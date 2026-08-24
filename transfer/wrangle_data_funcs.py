import pandas as pd
import requests
import unidecode
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def import_data_from_vastaav(year, n_gws):
    year_range = f'20{year-1}-{year}'
    _local_root = Path(__file__).parent / "outputs" / "scraped_data" / "data"
    gw_df_list = []
    for i in range(1, n_gws+1):
        local_path = _local_root / year_range / "gws" / f"gw{i}.csv"
        if local_path.exists():
            gw_df = pd.read_csv(local_path, index_col=0)
        else:
            gw_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{year_range}/gws/gw{i}.csv"
            gw_df = pd.read_csv(gw_url, index_col=0)
        gw_df['gw'] = i
        gw_df_list.append(gw_df)

    gw_df = pd.concat(gw_df_list)
    return gw_df.reset_index()

def get_team_goals(was_home, team_h_score, team_a_score):
    if was_home:
        return team_h_score
    else:
        return team_a_score
def get_opponent_goals(was_home, team_h_score, team_a_score):
    if was_home:
        return team_a_score
    else:
        return team_h_score
def get_team_points(was_home, team_h_score, team_a_score):
    if was_home:
        return 3 if team_h_score > team_a_score else 0 if team_h_score < team_a_score else 1
    else:
        return 3 if team_a_score > team_h_score else 0 if team_a_score < team_h_score else 1
def get_opponent_points(team_points):
    if team_points == 3:
        return 0
    elif team_points == 0:
        return 3
    else:
        return 1

def add_team_data(gw_df):
    gw_df['team_goals'] = gw_df.apply(lambda row: get_team_goals(row['was_home'], row['team_h_score'], row['team_a_score']), axis=1)
    gw_df['opponent_goals'] = gw_df.apply(lambda row: get_opponent_goals(row['was_home'], row['team_h_score'], row['team_a_score']), axis=1)
    gw_df['team_points'] = gw_df.apply(lambda row: get_team_points(row['was_home'], row['team_h_score'], row['team_a_score']), axis=1)
    gw_df['opponent_points'] = gw_df['team_points'].apply(get_opponent_points)
    return gw_df

def combine_names(first_name, second_name):
    full_name = first_name + '_' + second_name    
    return full_name

def clean_name(name):
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = unidecode.unidecode(name)
    return name.strip().lower()

def ewma(gw_df, groupby_col, value_cols, alpha, rename_dict, remerge_cols):
    # Ensure the DataFrame is sorted by 'full_name' and 'gw'
    gw_df_sorted = gw_df.sort_values([groupby_col, 'gw']).reset_index()

    # Apply EWMA within each group
    ewma_gw_df = (
        gw_df_sorted
        .groupby(groupby_col, group_keys=False)
        [value_cols]
        .apply(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    )
    ewma_gw_df.rename(columns=rename_dict, inplace=True)
    ewma_gw_df = gw_df_sorted[remerge_cols].join(ewma_gw_df)
    return ewma_gw_df

def roll(gw_df, groupby_col, value_cols, rename_dict, remerge_cols, rolling_gws):
    # Ensure the DataFrame is sorted by 'full_name' and 'gw'
    gw_df_sorted = gw_df.sort_values([groupby_col, 'gw']).reset_index()

    # Apply rolling mean within each group
    rolling_gw_df = (
        gw_df_sorted
        .groupby(groupby_col, group_keys=False)
        [value_cols]
        .apply(lambda x: x.rolling(window=rolling_gws, min_periods=1).mean())
    )
    rolling_gw_df.rename(columns=rename_dict, inplace=True)
    rolling_gw_df = gw_df_sorted[remerge_cols].join(rolling_gw_df)
    return rolling_gw_df

def get_teams_df(gw_df):
    gw_df_teams = gw_df[['team', 'gw', 'team_goals', 'team_points']]
    gw_df_teams = gw_df_teams.groupby(['team', 'gw']).first().reset_index()
    return gw_df_teams

def get_teamcodes(year):
    teams_url = f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/20{year-1}-{year}/teams.csv'
    teams = pd.read_csv(teams_url)
    teamcode_dict = dict(zip(teams['id'], teams['name']))
    return teamcode_dict

def get_team_strength_df(year):
    """Season-level FPL team strength ratings, indexed by team name.

    Always fetched fresh from vaastav's GitHub repo, same as get_teamcodes() (its
    paired function, also used to name-map opponent_team below) -- deliberately NOT
    using a locally-cached teams.csv the way import_data_from_vastaav does for
    per-gameweek data: a local teams.csv can go stale (e.g. still listing a
    relegated team) if it isn't rescraped after squads/promotions change, which would
    silently produce NaN strength values for any current team missing from it.
    These ratings are a single per-season snapshot (not per-gameweek), which is fine
    since they're meant to change slowly. strength_attack_*/strength_defence_* can be
    0 pre-season/early in a season, before FPL has computed them.
    """
    year_range = f'20{year-1}-{year}'
    teams_url = f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/{year_range}/teams.csv'
    teams_df = pd.read_csv(teams_url)

    strength_cols = ['name', 'strength_overall_home', 'strength_overall_away',
                      'strength_attack_home', 'strength_attack_away',
                      'strength_defence_home', 'strength_defence_away']
    return teams_df[strength_cols].set_index('name')

def add_team_strength_features(gw_df, year):
    """Adds venue-appropriate own-team and opponent strength columns to gw_df.

    Uses each row's own team + venue (was_home) to pick that team's home/away rating,
    and the opponent's venue is the mirror image (they're playing the opposite venue).
    """
    strength_df = get_team_strength_df(year)
    teamcode_dict = get_teamcodes(year)
    opponent_name = gw_df['opponent_team'].map(teamcode_dict)
    was_home = (gw_df['was_home'] == True) | (gw_df['was_home'].astype(str) == 'True')

    stat_cols = {
        'overall': ('strength_overall_home', 'strength_overall_away'),
        'attack': ('strength_attack_home', 'strength_attack_away'),
        'defence': ('strength_defence_home', 'strength_defence_away'),
    }
    for stat, (home_col, away_col) in stat_cols.items():
        team_home_vals = gw_df['team'].map(strength_df[home_col])
        team_away_vals = gw_df['team'].map(strength_df[away_col])
        gw_df[f'team_strength_{stat}'] = team_home_vals.where(was_home, team_away_vals)

        # Opponent plays at the opposite venue from the player's own team.
        opp_home_vals = opponent_name.map(strength_df[home_col])
        opp_away_vals = opponent_name.map(strength_df[away_col])
        gw_df[f'opponent_strength_{stat}'] = opp_away_vals.where(was_home, opp_home_vals)

    return gw_df

def merge_ewma_dfs(ewma_gw_df_players, ewma_gw_df_teams, year):
    teamcode_dict = get_teamcodes(year)
    ewma_gw_df_players['opponent_team_name'] = ewma_gw_df_players['opponent_team'].map(teamcode_dict)
    ewma_gw_df = ewma_gw_df_players.merge(ewma_gw_df_teams, on=['team', 'gw'], how='left')

    ewma_gw_df_opponent_team = ewma_gw_df_teams.rename(columns={'team': 'opponent_team_name', 
                                                            'ewma_team_goals': 'ewma_team_goals_nw_opponent',
                                                            'ewma_team_points': 'ewma_team_points_nw_opponent'})
    ewma_gw_df = ewma_gw_df.merge(ewma_gw_df_opponent_team, on=['opponent_team_name', 'gw'], how='left')
    return ewma_gw_df

def lag_feature(ewma_gw_df, feature):
    new_feature = (
    ewma_gw_df
    .sort_values(['full_name', 'gw'])
    .groupby('full_name')[feature]
    .shift(-1))
    return new_feature

def get_ewma_df(year, gw, ewma_alpha):
    gw_df = import_data_from_vastaav(year, gw)
    gw_df = add_team_data(gw_df)

    gw_df['full_name'] = gw_df['name'].apply(clean_name)
    gw_df = add_team_strength_features(gw_df, year)

    player_value_cols = ['xP', 'assists', 'bonus', 'bps',
       'clean_sheets', 'creativity', 'expected_assists',
       'expected_goal_involvements', 'expected_goals',
       'expected_goals_conceded', 'goals_conceded', 'goals_scored',
       'ict_index', 'influence', 'minutes',
       'own_goals', 'penalties_missed', 'penalties_saved',
       'red_cards', 'saves', 'starts',
       'threat', 'total_points', 'transfers_balance',
       'transfers_in', 'transfers_out', 'value', 'yellow_cards']
    # Strength ratings are already a stable season-level number, not a noisy per-gw
    # stat, so they're carried through unmodified rather than EWMA'd/rolled.
    strength_cols = ['team_strength_attack', 'team_strength_defence', 'team_strength_overall',
                      'opponent_strength_attack', 'opponent_strength_defence', 'opponent_strength_overall']
    merge_cols_players = ['full_name', 'gw', 'total_points', 'position', 'team', 'opponent_team'] + strength_cols
    ewma_gw_df_players = ewma(gw_df, 'full_name', player_value_cols, ewma_alpha, {'total_points': 'ewma_total_points'}, merge_cols_players)

    gw_df_teams = get_teams_df(gw_df)
    team_value_cols = ['team_goals', 'team_points']
    merge_cols_teams = ['team', 'gw']
    ewma_gw_df_teams = ewma(gw_df_teams, 'team', team_value_cols, ewma_alpha, {'team_goals': 'ewma_team_goals',
                                                                            'team_points': 'ewma_team_points'}, merge_cols_teams)
    merged_ewma_df = merge_ewma_dfs(ewma_gw_df_players, ewma_gw_df_teams, year)
    return merged_ewma_df

def get_rolling_df(year, gw, rolling_gws):
    gw_df = import_data_from_vastaav(year, gw)
    gw_df = add_team_data(gw_df)

    gw_df['full_name'] = gw_df['name'].apply(clean_name)
    gw_df = add_team_strength_features(gw_df, year)

    player_value_cols = ['xP', 'assists', 'bonus', 'bps',
       'clean_sheets', 'creativity', 'expected_assists',
       'expected_goal_involvements', 'expected_goals',
       'expected_goals_conceded', 'goals_conceded', 'goals_scored',
       'ict_index', 'influence', 'minutes',
       'own_goals', 'penalties_missed', 'penalties_saved',
       'red_cards', 'saves', 'starts',
       'threat', 'total_points', 'transfers_balance',
       'transfers_in', 'transfers_out', 'value', 'yellow_cards']
    # Strength ratings are already a stable season-level number, not a noisy per-gw
    # stat, so they're carried through unmodified rather than EWMA'd/rolled.
    strength_cols = ['team_strength_attack', 'team_strength_defence', 'team_strength_overall',
                      'opponent_strength_attack', 'opponent_strength_defence', 'opponent_strength_overall']
    merge_cols_players = ['full_name', 'gw', 'total_points', 'position', 'team', 'opponent_team'] + strength_cols
    ewma_gw_df_players = roll(gw_df, 'full_name', player_value_cols, {'total_points': 'ewma_total_points'}, merge_cols_players, rolling_gws)

    gw_df_teams = get_teams_df(gw_df)
    team_value_cols = ['team_goals', 'team_points']
    merge_cols_teams = ['team', 'gw']
    ewma_gw_df_teams = roll(gw_df_teams, 'team', team_value_cols, {'team_goals': 'ewma_team_goals',
                                                                            'team_points': 'ewma_team_points'}, merge_cols_teams, rolling_gws)
    merged_ewma_df = merge_ewma_dfs(ewma_gw_df_players, ewma_gw_df_teams, year)
    return merged_ewma_df

def lag_data_for_training(merged_ewma_df):
    merged_ewma_df['total_points_nw'] = lag_feature(merged_ewma_df, 'total_points')
    merged_ewma_df['opponent_nw'] = lag_feature(merged_ewma_df, 'opponent_team_name')
    return merged_ewma_df

def scale_df(df, features):
    scaler = StandardScaler()
    scaler.fit(df[features])
    df[features] = scaler.transform(df[features])
    return df, scaler

def get_fixture_dict(gw, year):
    teamcode_dict = get_teamcodes(year)
    fixtures_url = 'https://fantasy.premierleague.com/api/fixtures/'
    r = requests.get(fixtures_url, timeout=20).json()
    fixtures_df = pd.json_normalize(r)
    fixtures_df_gw = fixtures_df.query(f'event=={gw}')
    h_teams = fixtures_df_gw['team_h'].map(teamcode_dict)
    a_teams = fixtures_df_gw['team_a'].map(teamcode_dict)
    fixture_dict = {**dict(zip(h_teams, a_teams)), **dict(zip(a_teams, h_teams))}
    return fixture_dict

def get_gw_df(gw, year):
    gw_df = import_data_from_vastaav(year, gw)
    gw_df = add_team_data(gw_df)
    gw_df['full_name'] = gw_df['name'].apply(clean_name)
    return gw_df

def get_fpl_points_scored_df(gw_df, year):
    points_scored_groupby_cols = ['gw', 'team', 'position']
    points_scored_df = gw_df.groupby(points_scored_groupby_cols
                                            ).sum(numeric_only=True)[['total_points', 'minutes']]

    points_scored_df['total_points'] = points_scored_df['total_points'] / (points_scored_df['minutes'] / 90)
    points_scored_df = points_scored_df.drop('minutes', axis=1)

    points_scored_df_combined = pd.DataFrame(index=points_scored_df.loc[:,:, 'DEF'].index)
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        points_scored_df_pos = points_scored_df.loc[:,:, pos]
        points_scored_df_pos = points_scored_df_pos.rename(columns={'total_points': f'points_scored_{pos}'})
        points_scored_df_combined = points_scored_df_combined.merge(points_scored_df_pos, left_index=True, right_index=True)

    return points_scored_df_combined.reset_index()[['team', 'gw'] + [f'points_scored_{pos}' for pos in ['GK', 'DEF', 'MID', 'FWD']]]

def get_fpl_points_conceded_df(gw_df, year):
    points_conceded_groupby_cols = ['gw', 'opponent_team', 'position']
    points_conceded_df = gw_df.groupby(points_conceded_groupby_cols
                                            ).sum(numeric_only=True)[['total_points', 'minutes']]
    
    points_conceded_df['total_points'] = points_conceded_df['total_points'] / (points_conceded_df['minutes'] / 90)
    points_conceded_df = points_conceded_df.drop('minutes', axis=1)

    points_conceded_df_combined = pd.DataFrame(index=points_conceded_df.loc[:,:, 'DEF'].index)
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        points_conceded_df_pos = points_conceded_df.loc[:,:, pos]
        points_conceded_df_pos = points_conceded_df_pos.rename(columns={'total_points': f'points_conceded_{pos}'})
        points_conceded_df_combined = points_conceded_df_combined.merge(points_conceded_df_pos, left_index=True, right_index=True)

    teamcode_dict = get_teamcodes(year)
    points_conceded_df_combined = points_conceded_df_combined.reset_index(names=['gw', 'team'])
    points_conceded_df_combined['team'] = points_conceded_df_combined['team'].map(teamcode_dict)
    return points_conceded_df_combined[['team', 'gw'] + [f'points_conceded_{pos}' for pos in ['GK', 'DEF', 'MID', 'FWD']]]

def get_fpl_points_by_team(year, gw, n_gws=10):
    gw_df = get_gw_df(gw-1, year)
    points_scored_gw_df = get_fpl_points_scored_df(gw_df, year)
    points_conceded_gw_df = get_fpl_points_conceded_df(gw_df, year).rename(columns={'team': 'opponent_team'})

    points_scored_rolled = roll(points_scored_gw_df, 'team', 
        ['points_scored_GK', 'points_scored_DEF', 'points_scored_MID', 'points_scored_FWD'],
        {'points_scored_GK': 'avg_points_scored_GK', 'points_scored_DEF': 'avg_points_scored_DEF',
        'points_scored_MID': 'avg_points_scored_MID', 'points_scored_FWD': 'avg_points_scored_FWD'}, 
        ['team', 'gw'], n_gws)

    points_conceded_rolled = roll(points_conceded_gw_df, 'opponent_team', 
        ['points_conceded_GK', 'points_conceded_DEF', 'points_conceded_MID', 'points_conceded_FWD'],
        {'points_conceded_GK': 'avg_points_conceded_GK_opponent', 'points_conceded_DEF': 'avg_points_conceded_DEF_opponent',
        'points_conceded_MID': 'avg_points_conceded_MID_opponent', 'points_conceded_FWD': 'avg_points_conceded_FWD_opponent'}, 
        ['opponent_team', 'gw'], n_gws)

    points_scored_rolled_gw = points_scored_rolled.query(f'gw=={gw-1}')
    points_conceded_rolled_gw = points_conceded_rolled.query(f'gw=={gw-1}')
    fixture_dict = get_fixture_dict(gw, year)
    points_conceded_rolled_gw['team'] = points_conceded_rolled_gw['opponent_team'].map(fixture_dict)

    fpl_points_by_team = points_conceded_rolled_gw.merge(points_scored_rolled_gw, left_on=['team', 'gw'], 
                                                        right_on=['team', 'gw'], how='left')
    return fpl_points_by_team.set_index(['team', 'gw', 'opponent_team'])

def get_fixture_diff_index(fpl_points_by_team):
    multiplier_cols = []
    for col in fpl_points_by_team.columns:
        median_score = fpl_points_by_team[col].median()
        fpl_points_by_team[f'{col}_multiplier'] = fpl_points_by_team[col] / median_score
        multiplier_cols.append(f'{col}_multiplier')

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        fpl_points_by_team[f'fixture_diff_{pos}_multiplier'] = (
            fpl_points_by_team[f'avg_points_conceded_{pos}_opponent_multiplier']
            * fpl_points_by_team[f'avg_points_scored_{pos}_multiplier']
        ).clip(lower=0.5, upper=2)
    
    return fpl_points_by_team

def integrate_fixture_diff_index(pred_df, fixture_diff_index):
    fixture_diff_index_multiplier = fixture_diff_index[[f'fixture_diff_{pos}_multiplier' for pos in ['GK', 'DEF', 'MID', 'FWD']]]
    fixture_diff_index_multiplier = fixture_diff_index_multiplier.rename(columns={f'fixture_diff_{pos}_multiplier': pos for pos in ['GK', 'DEF', 'MID', 'FWD']})
    fixture_diff_index_multiplier_melt = fixture_diff_index_multiplier.reset_index().drop(
        ['gw', 'opponent_team'], axis=1).melt(id_vars='team', var_name='position', value_name='fixture_diff_index')
    pred_df = pred_df.merge(fixture_diff_index_multiplier_melt, on=['team', 'position'], how='left')
    pred_df['predicted_points_adj'] = pred_df['predicted_points'] * pred_df['fixture_diff_index']
    return pred_df