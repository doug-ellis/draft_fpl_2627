import pandas as pd

POS_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def find_choices(pred_df):
    return pred_df.query('is_available==True').sort_values('y_pred', ascending=False).groupby('pos').head(1)

def choose_player2(choices, npicks_before_next):
    """    Function to choose players based on predicted deltas. """
    jed_algorithm_dict = {}
    if npicks_before_next == 0:
        npicks_before_next = 1
    n_deltas = sum(1 for c in choices.columns if c.startswith("y_pred_delta_"))
    npicks_before_next = min(npicks_before_next, n_deltas)
    for choice in choices.index:
        deltas_scaled = []
        for pick in range(npicks_before_next):
            delta = choices.loc[choice, f'y_pred_delta_{pick+1}']
            delta_scaled = (1/4)**pick * delta
            deltas_scaled.append(delta_scaled)
        jed_algorithm_dict[choice] = deltas_scaled

    jed_algorithm_df = pd.DataFrame(jed_algorithm_dict).T
    jed_algorithm_df['sum'] = jed_algorithm_df.sum(axis=1)
    return jed_algorithm_df['sum'].idxmax()

def get_eligible_players(user_id, main_df):
    """    Function to get eligible players for a user based on club constraints. """ 
    user_team_counts = main_df.query('picked_by == @user_id')['team_name'].value_counts()
    used_teams = user_team_counts[user_team_counts==3].index.to_list()
    eligible_players = main_df.query('is_available == True and team_name not in @used_teams')
    return eligible_players

def get_choices(user_id, eligible_players, main_df):
    """    Function to get choices for a user based on club constraints and position quotas. """
    choices = find_choices(eligible_players)
    user_pos_counts = main_df.query('picked_by == @user_id')['pos'].value_counts()
    filled_positions = []
    for pos, count in POS_QUOTA.items():
        if user_pos_counts.get(pos, 0) == count:
            filled_positions.append(pos)
    choices_filtered = choices.query('pos not in @filled_positions')
    return choices_filtered

def record_choice(user_teams_dict, user_id, pick, main_df, i):
    user_teams_dict[user_id].append(pick)
    main_df.loc[pick, 'is_available'] = False
    main_df.loc[pick, 'pick'] = i + 1
    main_df.loc[pick, 'picked_by'] = user_id

    return main_df, user_teams_dict

def get_positions_needed_in_formation(user_pos_counts, picks_remaining):

    eligible_formations = [
    {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},  # 1352
    {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},  # 1433
    {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},  # 1442
    {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},  # 1451
    {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},  # 1343
    {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},  # 1532
    {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},  # 1541
    ]
    
    allowed_positions = []
    for formation in eligible_formations:
        needed = {
            pos: formation[pos] - user_pos_counts.get(pos, 0)
            for pos in formation
        }

        if all(need >= 0 for need in needed.values()) and (sum(needed.values()) == picks_remaining):
            allowed_positions = allowed_positions + [pos for pos, need in needed.items() if need > 0]

    allowed_positions = list(set(allowed_positions))
    return allowed_positions