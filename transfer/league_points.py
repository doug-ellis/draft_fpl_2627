import requests
import pandas as pd
from squad_selection import get_owner_dict


def get_current_gw():
    """Real 'as of today' gameweek from the draft league's bootstrap-static feed --
    distinct from a prediction file's TARGET_GW/latest_gw, which reflects the
    prediction-file horizon (see WORKFLOW.md), not the calendar."""
    resp = requests.get('https://draft.premierleague.com/api/bootstrap-static', timeout=20)
    resp.raise_for_status()
    return resp.json()['events']['current']


def load_owner_actual_points(league_id=3875):
    """Real points each owner has already scored, from the draft league's H2H
    matches (each owner appears in exactly one match per GW). For each such match,
    returns both:
      - actual_points: that owner's raw fantasy score for the week, independent of
        win/loss.
      - actual_league_points: the real H2H result for that week, 3/1/0 for
        win/draw/loss -- derived by comparing the two sides' points, since the
        matches feed doesn't expose a win/loss field directly. Cross-check this
        against the live `standings[].total` via load_standings() below.
    Only includes GWs that have started or finished -- future GWs are 0-0
    placeholders in this feed, and the current GW's points update live as matches
    are played. Each row also carries `finished`, since the live standings total
    (see load_standings()) only reflects finished matches -- a mid-week sanity
    check needs to compare against finished rows only, not the live in-progress GW."""
    resp = requests.get(f'https://draft.premierleague.com/api/league/{league_id}/details', timeout=20)
    resp.raise_for_status()
    data = resp.json()

    entry_id_by_league_entry_id = {entry['id']: entry['entry_id'] for entry in data['league_entries']}
    owner_dict = get_owner_dict()

    rows = []
    for match in data['matches']:
        if not (match['started'] or match['finished']):
            continue
        for side, other_side in (('1', '2'), ('2', '1')):
            entry_id = entry_id_by_league_entry_id.get(match[f'league_entry_{side}'])
            owner_name = owner_dict.get(float(entry_id)) if entry_id is not None else None
            if owner_name is None:
                continue
            own_points = match[f'league_entry_{side}_points']
            opp_points = match[f'league_entry_{other_side}_points']
            if own_points > opp_points:
                actual_league_points = 3
            elif own_points == opp_points:
                actual_league_points = 1
            else:
                actual_league_points = 0
            rows.append({
                'gw': match['event'],
                'owner_name': owner_name,
                'actual_points': own_points,
                'actual_league_points': actual_league_points,
                'finished': match['finished'],
            })
    return pd.DataFrame(rows)


def load_standings(league_id=3875):
    """Live cumulative standings straight from the API (rank, total league points,
    W/D/L) -- used only to sanity-check the actual_league_points reconstruction in
    load_owner_actual_points() above, since that's derived rather than read from a
    dedicated field."""
    resp = requests.get(f'https://draft.premierleague.com/api/league/{league_id}/details', timeout=20)
    resp.raise_for_status()
    data = resp.json()

    entry_id_by_league_entry_id = {entry['id']: entry['entry_id'] for entry in data['league_entries']}
    owner_dict = get_owner_dict()

    rows = []
    for standing in data['standings']:
        entry_id = entry_id_by_league_entry_id.get(standing['league_entry'])
        owner_name = owner_dict.get(float(entry_id)) if entry_id is not None else None
        if owner_name is None:
            continue
        rows.append({
            'owner_name': owner_name,
            'rank': standing['rank'],
            'total': standing['total'],
            'matches_won': standing['matches_won'],
            'matches_drawn': standing['matches_drawn'],
            'matches_lost': standing['matches_lost'],
        })
    return pd.DataFrame(rows)


def compute_expected_league_points(actual_points_df):
    """Per-GW, ranks owners by actual_points (1=highest) and awards a linear share
    of 3 points down to 0 -- 3 * (n_owners - rank) / (n_owners - 1) -- mirroring
    real match scoring (3 win / 1 draw / 0 loss) but spread across the whole league
    each week instead of just your one assigned H2H opponent. Ties split the
    spanned ranks' values evenly (pandas rank(method='average'))."""
    df = actual_points_df.copy()
    df['rank'] = df.groupby('gw')['actual_points'].rank(method='average', ascending=False)
    n_owners = df.groupby('gw')['owner_name'].transform('count')
    df['expected_league_points'] = 3 * (n_owners - df['rank']) / (n_owners - 1)
    return df
