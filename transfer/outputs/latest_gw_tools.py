from pathlib import Path
import re

import pandas as pd


def _extract_gw(path_obj):
    match = re.search(r"predicted_gw(\d+)_simple\.csv$", path_obj.name)
    if not match:
        return None
    return int(match.group(1))


def get_latest_gw(predictions_dir="predictions"):
    predictions_path = Path(predictions_dir)
    simple_files = predictions_path.glob("predicted_gw*_simple.csv")
    gws = [gw for gw in (_extract_gw(path_obj) for path_obj in simple_files) if gw is not None]
    if not gws:
        raise FileNotFoundError(f"No predicted_gw*_simple.csv files found in {predictions_path.resolve()}")
    return max(gws)


def load_gw_outputs(gw, predictions_dir="predictions", fixture_dir="fixture_difficulty"):
    predictions_path = Path(predictions_dir)
    fixture_path = Path(fixture_dir)
    pred_simple = pd.read_csv(predictions_path / f"predicted_gw{gw}_simple.csv")
    pred_full = pd.read_csv(predictions_path / f"predicted_gw{gw}.csv")
    fixture_diff = pd.read_csv(fixture_path / f"fixture_difficulty_gw{gw}.csv")
    return pred_simple, pred_full, fixture_diff


def load_latest_outputs(predictions_dir="predictions", fixture_dir="fixture_difficulty"):
    latest_gw = get_latest_gw(predictions_dir=predictions_dir)
    pred_simple, pred_full, fixture_diff = load_gw_outputs(
        latest_gw,
        predictions_dir=predictions_dir,
        fixture_dir=fixture_dir,
    )
    return latest_gw, pred_simple, pred_full, fixture_diff
