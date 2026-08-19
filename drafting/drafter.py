import argparse
from pathlib import Path

import pandas as pd
from drafting_funcs import (
    choose_player2,
    get_choices,
    get_eligible_players,
    get_positions_needed_in_formation,
    record_choice,
)

PREDICTIONS_FILE = Path(__file__).parent / "player_score_predictions_delta_manAdj.csv"
TOP_N_PER_POS = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive FPL draft assistant.")
    parser.add_argument("--user-team", type=int, default=3, help="Your team number in the draft order.")
    parser.add_argument("--num-teams", type=int, default=10, help="Total number of teams in the draft.")
    return parser.parse_args()


def build_pick_order(nteams):
    fwd = list(range(1, nteams + 1))
    rev = list(range(nteams, 0, -1))
    return (fwd + rev) * 7 + fwd


def resolve_player(raw, valid_index):
    """Case-insensitive partial match; returns None if ambiguous or not found."""
    if raw in valid_index:
        return raw
    lower = raw.lower()
    matches = [n for n in valid_index if lower in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"  Ambiguous ({len(matches)} matches): {matches[:8]}")
    return None


def show_header(pick_num, total, user_id, user_player):
    flag = "  <-- YOUR TURN" if user_id == user_player else ""
    print(f"\n{'=' * 55}")
    print(f"  Pick {pick_num}/{total}  |  Team {user_id}{flag}")
    print(f"{'=' * 55}")


def show_your_squad(main_df, user_player):
    squad = main_df[main_df["picked_by"] == user_player][["pos", "team_name"]]
    if squad.empty:
        print("  Your squad: (empty)")
        return
    print("  Your squad so far:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        for name, row in squad[squad["pos"] == pos].iterrows():
            print(f"    {pos:3s}  {name:30s}  {row['team_name']}")


def show_candidates(eligible_players, open_positions, recommendation):
    _print_player_table(
        eligible_players[eligible_players["pos"].isin(open_positions)],
        recommendation=recommendation,
    )


def show_top_available(main_df):
    _print_player_table(main_df[main_df["is_available"] == True])


def _print_player_table(players, recommendation=None):
    print(f"\n  {'Name':<30}  {'Pos':<4}  {'Team':<20}  {'Pred':>6}")
    print(f"  {'-'*30}  {'-'*4}  {'-'*20}  {'-'*6}")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        top = players[players["pos"] == pos].sort_values("y_pred", ascending=False).head(TOP_N_PER_POS)
        for name, row in top.iterrows():
            marker = "  <--" if name == recommendation else ""
            print(f"  {name:<30}  {pos:<4}  {row['team_name']:<20}  {row['y_pred']:>6.1f}{marker}")


def auto_pick(choices, eligible_players, npicks_before_next):
    """Fall back to best eligible player when all position quotas are filled (bench picks)."""
    if choices.empty:
        if eligible_players.empty:
            raise RuntimeError("No eligible players remain — check club/position constraints.")
        return eligible_players.sort_values("y_pred", ascending=False).index[0]
    return choose_player2(choices, npicks_before_next)


def your_turn(choices, eligible_players, npicks_before_next):
    recommendation = auto_pick(choices, eligible_players, npicks_before_next)
    show_candidates(eligible_players, choices["pos"].unique().tolist(), recommendation)
    print(f"\n  Recommended: {recommendation}")
    response = input("  Accept? (Enter/y) or type a player name: ").strip()
    if response in ("", "y", "Y"):
        return recommendation
    resolved = resolve_player(response, eligible_players.index)
    while resolved is None:
        print("  Not found. Try a partial name, e.g. 'Salah'.")
        response = input("  Player name: ").strip()
        resolved = resolve_player(response, eligible_players.index)
    return resolved


def opponent_turn(main_df, choices, eligible_players, npicks_before_next, user_id):
    show_top_available(main_df)
    available = main_df[main_df["is_available"] == True].index
    response = input(f"\n  Team {user_id} picked (Enter to auto): ").strip()
    if response == "":
        return auto_pick(choices, eligible_players, npicks_before_next)
    resolved = resolve_player(response, available)
    while resolved is None:
        print("  Not in available players. Try again.")
        response = input(f"  Team {user_id} picked: ").strip()
        resolved = resolve_player(response, available)
    return resolved


def show_final_squad(main_df, user_player):
    squad = main_df[main_df["picked_by"] == user_player][["pos", "team_name", "pick"]].copy()
    squad["pick"] = squad["pick"].astype(int)
    print(f"\n{'=' * 55}")
    print("  YOUR FINAL SQUAD")
    print(f"{'=' * 55}")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        for name, row in squad[squad["pos"] == pos].iterrows():
            print(f"  {pos:3s}  {name:30s}  {row['team_name']:<20}  (pick {row['pick']})")
    print(f"{'=' * 55}")


def main():
    args = parse_args()
    user_player = args.user_team
    nteams = args.num_teams

    main_df = pd.read_csv(PREDICTIONS_FILE, index_col="full_name")
    main_df["picked_by"] = None

    user_teams_dict = {i + 1: [] for i in range(nteams)}
    pick_order = build_pick_order(nteams)
    total_picks = len(pick_order)

    for i, user_id in enumerate(pick_order):
        if i > total_picks - nteams - 1:
            npicks_before_next = 0
        else:
            try:
                next_pick = pick_order.index(user_id, i + 1, total_picks)
                npicks_before_next = (next_pick - i) - 1
            except ValueError:
                npicks_before_next = 0

        eligible_players = get_eligible_players(user_id, main_df)
        choices = get_choices(user_id, eligible_players, main_df)
        user_pos_counts = main_df.query("picked_by == @user_id")["pos"].value_counts()
        picks_remaining_for11 = 11 - len(user_teams_dict[user_id])

        if picks_remaining_for11 > 0:
            allowed_positions = get_positions_needed_in_formation(user_pos_counts, picks_remaining_for11)
            choices = choices.query("pos in @allowed_positions")

        show_header(i + 1, total_picks, user_id, user_player)

        if user_id == user_player:
            show_your_squad(main_df, user_player)
            pick = your_turn(choices, eligible_players, npicks_before_next)
        else:
            pick = opponent_turn(main_df, choices, eligible_players, npicks_before_next, user_id)

        main_df, user_teams_dict = record_choice(user_teams_dict, user_id, pick, main_df, i)
        print(f"  >> Team {user_id} picks: {pick}")

    show_final_squad(main_df, user_player)
    output_path = Path(__file__).parent / "output" / "drafted_players.csv"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    main_df.to_csv(output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()