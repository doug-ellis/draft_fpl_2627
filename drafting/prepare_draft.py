import json
import sys
from pathlib import Path

import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from clean_data_funcs import combine_clean_names

FPL_API = "https://fantasy.premierleague.com/api/"
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
N_DELTAS = 18
OUTPUT_FILE = Path(__file__).parent / "player_score_predictions_delta_manAdj.csv"
INJURED_FILE = Path(__file__).parent / "injured_players.json"


def fetch_bootstrap():
    try:
        r = requests.get(FPL_API + "bootstrap-static/", timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to fetch FPL bootstrap data: {e}") from e
    return r.json()


def fetch_team_goals(team_id_to_name):
    try:
        r = requests.get(FPL_API + "fixtures/", timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to fetch FPL fixtures: {e}") from e
    fixtures = pd.json_normalize(r.json())
    finished = fixtures[fixtures["finished"] == True].copy()
    if finished.empty:
        # pre-season: no results yet, use zeros so model can still run
        names = list(team_id_to_name.values())
        return pd.DataFrame({"GF": 0, "GA": 0}, index=pd.Index(names, name="team_name"))
    gf = (
        finished.groupby("team_h")["team_h_score"].sum()
        .add(finished.groupby("team_a")["team_a_score"].sum(), fill_value=0)
        .rename(index=team_id_to_name)
    )
    ga = (
        finished.groupby("team_h")["team_a_score"].sum()
        .add(finished.groupby("team_a")["team_h_score"].sum(), fill_value=0)
        .rename(index=team_id_to_name)
    )
    df = pd.DataFrame({"GF": gf, "GA": ga})
    df.index.name = "team_name"
    return df


def build_player_df(bootstrap, team_goals_df):
    players = pd.json_normalize(bootstrap["elements"])
    teams = pd.json_normalize(bootstrap["teams"])
    team_id_to_name = dict(zip(teams["id"], teams["name"]))

    df = players[["first_name", "second_name", "now_cost", "team", "element_type", "total_points"]].copy()
    df = combine_clean_names(df, "first_name", "second_name")
    df["pos"] = df["element_type"].map(POSITION_MAP)
    df["team_name"] = df["team"].map(team_id_to_name)
    unmapped = df["team_name"].isna().sum()
    if unmapped:
        print(f"  Warning: {unmapped} players have unmapped team IDs.")
    df = df.merge(team_goals_df.reset_index(), on="team_name", how="left")
    df["is_available"] = True
    return df.set_index("full_name")


MODEL_FEATURES = ["now_cost", "GF", "GA"]


def fit_and_predict(df):
    parts = []
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_df = df[df["pos"] == pos].copy()
        train = pos_df[pos_df["total_points"] > 0].dropna(subset=MODEL_FEATURES)
        if len(train) < 2:
            print(f"  Warning: insufficient training data for {pos} ({len(train)} samples); using mean.")
            pos_df["y_pred"] = pos_df["total_points"].mean()
            parts.append(pos_df.dropna(subset=MODEL_FEATURES))
            continue
        model = LinearRegression()
        model.fit(train[MODEL_FEATURES], train["total_points"])
        pred_df = pos_df.dropna(subset=MODEL_FEATURES).copy()
        pred_df["y_pred"] = model.predict(pred_df[MODEL_FEATURES])
        parts.append(pred_df)
    return pd.concat(parts)


def compute_deltas(df):
    # delta_i[r] = y_pred[rank r+i-1] - y_pred[rank r+i]: how much value drops if you wait i picks
    parts = []
    delta_cols = [f"y_pred_delta_{i}" for i in range(1, N_DELTAS + 1)]
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_df = df[df["pos"] == pos].sort_values("y_pred", ascending=False).copy()
        pos_df = pos_df.reset_index()
        for i in range(1, N_DELTAS + 1):
            pos_df[f"y_pred_delta_{i}"] = pos_df["y_pred"].shift(-(i - 1)) - pos_df["y_pred"].shift(-i)
        # players near the bottom have no one below them; 0 is the correct delta
        pos_df[delta_cols] = pos_df[delta_cols].fillna(0)
        parts.append(pos_df)
    return pd.concat(parts).set_index("full_name")


def apply_injuries(df):
    try:
        injured = json.loads(INJURED_FILE.read_text())
    except FileNotFoundError:
        print("  Warning: injured_players.json not found; skipping injury adjustments.")
        return df
    except json.JSONDecodeError as e:
        print(f"  Warning: injured_players.json is malformed ({e}); skipping injury adjustments.")
        return df
    delta_cols = [c for c in df.columns if c.startswith("y_pred_delta_")]
    for name in injured:
        if name in df.index:
            df.loc[name, ["y_pred"] + delta_cols] = 0
    return df


def main():
    print("Fetching FPL bootstrap data...")
    bootstrap = fetch_bootstrap()
    teams = pd.json_normalize(bootstrap["teams"])
    team_id_to_name = dict(zip(teams["id"], teams["name"]))

    print("Fetching fixture results for team goals...")
    team_goals_df = fetch_team_goals(team_id_to_name)

    df = build_player_df(bootstrap, team_goals_df)

    print("Fitting models and predicting scores...")
    df = fit_and_predict(df)

    print("Computing pick-value deltas...")
    df = compute_deltas(df)

    print("Applying injury adjustments from injured_players.json...")
    df = apply_injuries(df)

    df.to_csv(OUTPUT_FILE)
    print(f"Saved {len(df)} players to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
