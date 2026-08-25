import argparse
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from urllib3.util.retry import Retry

from modelling_funcs import create_model, predict_scores
from wrangle_data_funcs import (
    attach_target_week_points_by_team,
    build_fpl_points_snapshot,
    clean_name,
    combine_names,
    get_ewma_df,
    get_fixture_dict,
    get_fixture_diff_index,
    get_fpl_points_by_team,
    get_rolling_df,
    integrate_fixture_diff_index,
    lag_data_for_training,
)

def get_training_df(training_years, n_gws, avg_type, alpha=0.3, rolling_gws=4):
    training_dfs = []
    for year in training_years:
        if avg_type=='ewma':
            gw_df = get_ewma_df(year, n_gws, alpha)
        elif avg_type=='rolling':
            gw_df = get_rolling_df(year, n_gws, rolling_gws)
        training_df = lag_data_for_training(gw_df).dropna(subset=['total_points_nw'])
        non_zero_players = training_df.groupby('full_name').sum().query('total_points_nw>0').index
        training_df_f = training_df.query('gw>10 and full_name in @non_zero_players')
        training_dfs.append(training_df_f.assign(year=year+2000))
    training_df_f = pd.concat(training_dfs, ignore_index=True)
    return training_df_f

def build_prediction_snapshot(year, anchor_gw, avg_type, alpha=0.3, rolling_gws=4):
    """Player/team 'current form' snapshot, anchored to real played data as of
    anchor_gw-1. Expensive (fetches/derives EWMA or rolling features) -- build once
    and reuse across every target week in a forecast horizon via attach_target_week,
    since it doesn't depend on which future week is being forecast (no real data
    exists yet for future weeks to re-derive form from)."""
    if avg_type=='ewma':
        full_df = get_ewma_df(year, anchor_gw-1, alpha).drop(['ewma_team_goals_nw_opponent', 'ewma_team_points_nw_opponent'], axis=1)
    elif avg_type=='rolling':
        full_df = get_rolling_df(year, anchor_gw-1, rolling_gws).drop(['ewma_team_goals_nw_opponent', 'ewma_team_points_nw_opponent'], axis=1)

    full_df = full_df.query(f'gw<={anchor_gw-1}')

    # Teams can blank in anchor_gw-1; keep each player's latest available snapshot up to anchor_gw-1.
    prediction_snapshot = (
        full_df
        .sort_values(['full_name', 'gw'])
        .groupby('full_name', as_index=False)
        .tail(1)
        .copy()
    )

    # Each team's own latest team-level snapshot, independent of which player it's read off of
    # (players on the same team can have different "latest available" gws after the tail(1) above).
    opp_team_df = (
        full_df[['team', 'gw', 'ewma_team_goals', 'ewma_team_points']]
        .sort_values(['team', 'gw'])
        .groupby('team', as_index=False)
        .tail(1)
        .drop(columns='gw')
    )
    return prediction_snapshot, opp_team_df

def attach_target_week(prediction_snapshot, opp_team_df, target_gw, year):
    """Cheap per-week step: attaches target_gw's fixture/opponent onto the fixed
    anchor snapshot (the FPL API already knows the full season's schedule in advance)."""
    prediction_df = prediction_snapshot.copy()
    fixture_dict = get_fixture_dict(target_gw, year)
    prediction_df['team_name_nw_opponent'] = prediction_df['team'].map(fixture_dict)
    prediction_df = prediction_df.merge(opp_team_df, left_on='team_name_nw_opponent', right_on='team', suffixes=('', '_nw_opponent'))
    return prediction_df

def get_prediction_df(year, gw, avg_type, alpha=0.3, rolling_gws=4):
    """Backward-compatible single-week wrapper."""
    prediction_snapshot, opp_team_df = build_prediction_snapshot(year, gw, avg_type, alpha, rolling_gws)
    return attach_target_week(prediction_snapshot, opp_team_df, gw, year)

def test_model(training_df_f, features, model_func):
    _, metrics_dict, _ = create_model(training_df_f, features, model_func, test=True)
    rmse_dict = {pos: (m['rmse'] if m else None) for pos, m in metrics_dict.items()}
    print(rmse_dict)
    return rmse_dict

def train_full_model(training_df, features, prediction_df, model_func):
    model_dict, _, scaler_dict = create_model(training_df, features, model_func, test=False)
    pred_df = predict_scores(prediction_df.dropna(subset=features).copy(), features, model_dict, scaler_dict)
    return pred_df

def build_retry_session(total_retries=3, backoff_factor=0.5):
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        allowed_methods=frozenset(["GET"]),
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=backoff_factor,
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session

def fetch_ownership_and_bootstrap(league_id, session):
    """One-time network fetch: league ownership + the draft bootstrap-static response
    (also used for its 'events' field to auto-detect end of season -- see
    get_end_of_season_gw). Reuse across every week in a forecast horizon rather than
    refetching per week; ownership doesn't vary week to week within one run."""
    league_url = f'https://draft.premierleague.com/api/league/{league_id}/element-status'
    r = session.get(league_url, timeout=20)
    r.raise_for_status()
    r = r.json()
    ownership_df = pd.json_normalize(r['element_status'])

    url = 'https://draft.premierleague.com/api/bootstrap-static'
    req = session.get(url, timeout=20)
    req.raise_for_status()
    req = req.json()
    players_df = pd.json_normalize(req['elements'])
    players_df['full_name'] = combine_names(players_df['first_name'], players_df['second_name']).apply(clean_name)
    merge_name_df = players_df[['id', 'full_name']]
    ownership_df_name = ownership_df.merge(merge_name_df, how='left', left_on='element', right_on='id')
    ownership_df_to_merge = ownership_df_name[['full_name', 'owner']]

    return ownership_df_to_merge, req['events']

def merge_ownership(pred_df, ownership_df_to_merge):
    """Cheap per-week merge against the already-fetched ownership table."""
    return pred_df.merge(ownership_df_to_merge, on='full_name', how='left')

def merge_ownership_data(pred_df, league_id, session):
    """Backward-compatible single-call wrapper."""
    ownership_df_to_merge, _events = fetch_ownership_and_bootstrap(league_id, session)
    return merge_ownership(pred_df, ownership_df_to_merge)

def get_end_of_season_gw(events):
    """events: the 'events' field from the draft bootstrap-static response, a dict
    shaped {"current": N, "data": [...]} (NOT a bare list, unlike classic FPL's
    bootstrap-static -- confirmed against the live API). Returns the last gameweek."""
    return max(e['id'] for e in events['data'])

# def get_fixture_difficulty_df(year, gw, n_gws):
#     gw_df = get_gw_df(gw-1, year)
#     points_conceded_gw_df = get_fpl_points_conceded_df(gw_df, year).rename(columns={'team': 'opponent_team'})

#     points_conceded_rolled = roll(points_conceded_gw_df, 'opponent_team', 
#         ['points_conceded_GK', 'points_conceded_DEF', 'points_conceded_MID', 'points_conceded_FWD'],
#         {'points_conceded_GK': 'avg_points_conceded_GK_opponent', 'points_conceded_DEF': 'avg_points_conceded_DEF_opponent',
#         'points_conceded_MID': 'avg_points_conceded_MID_opponent', 'points_conceded_FWD': 'avg_points_conceded_FWD_opponent'}, 
#         ['opponent_team', 'gw'], n_gws)

#     points_conceded_rolled_gw = points_conceded_rolled.query(f'gw=={gw-1}')
#     fixture_dict = get_fixture_dict(gw, year)
#     points_conceded_rolled_gw['team'] = points_conceded_rolled_gw['opponent_team'].map(fixture_dict)
#     points_conceded_rolled_gw.set_index('team')
#     return points_conceded_rolled_gw

def parse_args():
    parser = argparse.ArgumentParser(description="Predict FPL Draft scores for a target GW.")
    parser.add_argument("--training-years", nargs="+", type=int, default=[24, 25, 26], help="Training season suffixes, e.g. 24 25 26.")
    parser.add_argument("--training-n-gws", type=int, default=38, help="Number of GWs to include from each training year.")
    parser.add_argument("--pred-year", type=int, default=27, help="Prediction season suffix, e.g. 27 for 2026-27.")
    parser.add_argument("--pred-gw", type=int, required=True, help="Target gameweek to predict.")
    parser.add_argument("--end-gw", type=int, default=None, help="Last gameweek to forecast (inclusive). Default: auto-detected end of season.")
    parser.add_argument("--alpha", type=float, default=0.6, help="EWMA alpha if using ewma averaging.")
    parser.add_argument("--rolling-gws", type=int, default=4, help="Rolling window size if using rolling averaging.")
    parser.add_argument("--avg-type", choices=["rolling", "ewma"], default="ewma", help="Feature averaging strategy.")
    parser.add_argument("--model", choices=["elasticnet", "ridge", "lasso", "linear", "xgboost"], default="ridge", help="Model family.")
    parser.add_argument("--league-id", type=int, default=3875, help="Draft league ID for ownership pull.")
    parser.add_argument("--output-dir", default="outputs", help="Output folder under transfer/ unless absolute path is provided.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip train/test RMSE printout.")
    return parser.parse_args()

def get_features():
    return [
        'xP', 'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
        'expected_assists', 'expected_goal_involvements', 'expected_goals',
        'expected_goals_conceded', 'goals_conceded', 'goals_scored',
        'ict_index', 'influence', 'minutes', 'own_goals', 'penalties_missed',
        'penalties_saved', 'red_cards', 'saves', 'starts', 'threat',
        'ewma_total_points', 'value', 'yellow_cards',
        'ewma_team_goals', 'ewma_team_points', 'ewma_team_goals_nw_opponent',
        'ewma_team_points_nw_opponent',
    ]

def get_model_func(model_name):
    model_map = {
        "elasticnet": ElasticNet,
        "ridge": Ridge,
        "lasso": Lasso,
        "linear": LinearRegression,
        "xgboost": XGBRegressor,
    }
    return model_map[model_name]

def build_horizon_summary(simple_frames, horizon_gws):
    """simple_frames: dict of target_gw -> pred_df_simple (columns: full_name,
    position, team, predicted_points, predicted_points_adj, fixture_diff_index,
    owner), as accumulated in main()'s per-week loop. Builds one wide row per player
    across the whole horizon, entirely from already-computed in-memory frames (no
    disk re-read)."""
    base_cols = ['full_name', 'position', 'team', 'owner']
    per_gw_adj = {}
    identity_frames = []

    for target_gw in horizon_gws:
        df = simple_frames[target_gw]
        identity_frames.append(df[base_cols])
        per_gw_adj[target_gw] = df.set_index('full_name')['predicted_points_adj']

    # A player's position/team/owner can in principle shift between weeks (e.g. an
    # ownership change mid-horizon); use the LATEST week's values as the canonical
    # identity row.
    identity = pd.concat(identity_frames).drop_duplicates(subset='full_name', keep='last')

    horizon_df = identity.set_index('full_name')
    for target_gw in horizon_gws:
        horizon_df[f'gw{target_gw}_predicted_points_adj'] = per_gw_adj[target_gw]

    adj_cols = [f'gw{gw}_predicted_points_adj' for gw in horizon_gws]
    horizon_df['total_predicted_points_adj'] = horizon_df[adj_cols].sum(axis=1, skipna=True)
    horizon_df['n_gws_with_fixture'] = horizon_df[adj_cols].notna().sum(axis=1)
    horizon_df['avg_predicted_points_adj'] = (
        horizon_df['total_predicted_points_adj'] / horizon_df['n_gws_with_fixture'].replace(0, pd.NA)
    )

    out_cols = ['position', 'team', 'owner', 'total_predicted_points_adj',
                'avg_predicted_points_adj', 'n_gws_with_fixture'] + adj_cols
    horizon_df = horizon_df[out_cols].reset_index()
    horizon_df = horizon_df.sort_values('total_predicted_points_adj', ascending=False)
    return horizon_df

def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    prediction_output_dir = output_dir / "predictions"
    fixture_output_dir = output_dir / "fixture_difficulty"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    fixture_output_dir.mkdir(parents=True, exist_ok=True)

    features = get_features()
    model_func = get_model_func(args.model)
    session = build_retry_session()

    training_df = get_training_df(args.training_years, args.training_n_gws, args.avg_type, args.alpha, args.rolling_gws)
    if not args.skip_eval:
        _ = test_model(training_df, features, model_func)

    # --- build-once work: fit the model and anchor every "current form"/ownership
    # snapshot to the last real played week (args.pred_gw), regardless of horizon length ---
    model_dict, _, scaler_dict = create_model(training_df, features, model_func, test=False)
    prediction_snapshot, opp_team_df = build_prediction_snapshot(
        args.pred_year, args.pred_gw, args.avg_type, args.alpha, args.rolling_gws)
    points_scored_rolled_gw, points_conceded_rolled_gw = build_fpl_points_snapshot(
        args.pred_year, args.pred_gw, n_gws=10)
    ownership_df_to_merge, events = fetch_ownership_and_bootstrap(args.league_id, session)

    end_gw = args.end_gw if args.end_gw is not None else get_end_of_season_gw(events)
    if end_gw < args.pred_gw:
        raise ValueError(f"--end-gw ({end_gw}) is before --pred-gw ({args.pred_gw}).")
    horizon_gws = list(range(args.pred_gw, end_gw + 1))

    # --- per-week cheap work: only the fixture/opponent context varies ---
    simple_frames = {}
    for target_gw in horizon_gws:
        prediction_df = attach_target_week(prediction_snapshot, opp_team_df, target_gw, args.pred_year)
        pred_df = predict_scores(prediction_df.dropna(subset=features).copy(), features, model_dict, scaler_dict)
        pred_df = merge_ownership(pred_df, ownership_df_to_merge)

        fpl_points_by_team = attach_target_week_points_by_team(
            points_scored_rolled_gw, points_conceded_rolled_gw, target_gw, args.pred_year)
        fixture_diff_index = get_fixture_diff_index(fpl_points_by_team)
        pred_df = integrate_fixture_diff_index(pred_df, fixture_diff_index)

        pred_df_simple = pred_df[['full_name', 'position', 'team', 'predicted_points', 'predicted_points_adj', 'fixture_diff_index', 'owner']]
        pred_df.to_csv(prediction_output_dir / f"predicted_gw{target_gw}.csv", index=False)
        pred_df_simple.to_csv(prediction_output_dir / f'predicted_gw{target_gw}_simple.csv', index=False)
        fixture_diff_index.to_csv(fixture_output_dir / f'fixture_difficulty_gw{target_gw}.csv')

        simple_frames[target_gw] = pred_df_simple

    horizon_df = build_horizon_summary(simple_frames, horizon_gws)
    horizon_df.to_csv(
        prediction_output_dir / f"predicted_horizon_gw{horizon_gws[0]}_to_{horizon_gws[-1]}.csv", index=False)

    print(f"\nForecast {len(horizon_gws)} gameweek(s): {horizon_gws[0]} to {horizon_gws[-1]}.")

if __name__ == "__main__":
    main()