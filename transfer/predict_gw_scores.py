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

def get_prediction_df(year, gw, avg_type, alpha=0.3, rolling_gws=4):
    if avg_type=='ewma':
        full_df = get_ewma_df(year, gw-1, alpha).drop(['ewma_team_goals_nw_opponent', 'ewma_team_points_nw_opponent'], axis=1)
    elif avg_type=='rolling':
        full_df = get_rolling_df(year, gw-1, rolling_gws).drop(['ewma_team_goals_nw_opponent', 'ewma_team_points_nw_opponent'], axis=1)

    full_df = full_df.query(f'gw<={gw-1}')

    # Teams can blank in gw-1; keep each player's latest available snapshot up to gw-1.
    prediction_df = (
        full_df
        .sort_values(['full_name', 'gw'])
        .groupby('full_name', as_index=False)
        .tail(1)
        .copy()
    )

    fixture_dict = get_fixture_dict(gw, year)
    prediction_df['team_name_nw_opponent'] = prediction_df['team'].map(fixture_dict)

    # Each team's own latest team-level snapshot, independent of which player it's read off of
    # (players on the same team can have different "latest available" gws after the tail(1) above).
    opp_team_df = (
        full_df[['team', 'gw', 'ewma_team_goals', 'ewma_team_points']]
        .sort_values(['team', 'gw'])
        .groupby('team', as_index=False)
        .tail(1)
        .drop(columns='gw')
    )
    prediction_df = prediction_df.merge(opp_team_df, left_on='team_name_nw_opponent', right_on='team', suffixes=('', '_nw_opponent'))
    return prediction_df

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

def merge_ownership_data(pred_df, league_id, session):
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

    pred_df_owners = pred_df.merge(ownership_df_to_merge, on='full_name', how='left')
    return pred_df_owners

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
    parser.add_argument("--alpha", type=float, default=0.3, help="EWMA alpha if using ewma averaging.")
    parser.add_argument("--rolling-gws", type=int, default=4, help="Rolling window size if using rolling averaging.")
    parser.add_argument("--avg-type", choices=["rolling", "ewma"], default="rolling", help="Feature averaging strategy.")
    parser.add_argument("--model", choices=["elasticnet", "ridge", "lasso", "linear", "xgboost"], default="elasticnet", help="Model family.")
    parser.add_argument("--league-id", type=int, default=19188, help="Draft league ID for ownership pull.")
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
    prediction_df = get_prediction_df(args.pred_year, args.pred_gw, args.avg_type, args.alpha, args.rolling_gws)
    pred_df = train_full_model(training_df, features, prediction_df, model_func)
    pred_df = merge_ownership_data(pred_df, args.league_id, session)

    fpl_points_by_team = get_fpl_points_by_team(args.pred_year, args.pred_gw, n_gws=10)
    fixture_diff_index = get_fixture_diff_index(fpl_points_by_team)
    pred_df = integrate_fixture_diff_index(pred_df, fixture_diff_index)

    pred_df_simple = pred_df[['full_name', 'position', 'team', 'predicted_points', 'predicted_points_adj', 'fixture_diff_index', 'owner']]
    pred_df.to_csv(prediction_output_dir / f"predicted_gw{args.pred_gw}.csv", index=False)
    pred_df_simple.to_csv(prediction_output_dir / f'predicted_gw{args.pred_gw}_simple.csv', index=False)

    fixture_diff_index.to_csv(fixture_output_dir / f'fixture_difficulty_gw{args.pred_gw}.csv')

if __name__ == "__main__":
    main()