# Weekly GW Workflow

## 1) Run Predictions

From the repository root:

```powershell
python transfer/run_weekly_update.py --pred-gw 5 --pred-year 27
```

Optional flags:

- `--model elasticnet|ridge|lasso|linear|xgboost`
- `--skip-eval`

This writes outputs to:

- `transfer/outputs/predictions/predicted_gw<gw>.csv`
- `transfer/outputs/predictions/predicted_gw<gw>_simple.csv`
- `transfer/outputs/fixture_difficulty/fixture_difficulty_gw<gw>.csv`

## 2) Use One Notebook (No Copy/Paste)

Source modules (`latest_gw_tools.py`, `squad_selection.py`) live in `transfer/`; the generated notebooks live in `transfer/outputs/` and add `transfer/` to `sys.path` to import them. From `transfer/outputs`, load the latest outputs with:

```python
from latest_gw_tools import load_latest_outputs

latest_gw, pred_simple, pred_full, fixture_diff = load_latest_outputs()
print(latest_gw)
```

If you want a specific GW:

```python
from latest_gw_tools import load_gw_outputs

pred_simple, pred_full, fixture_diff = load_gw_outputs(35)
```

## 3) Evaluating Model / Feature Changes

Use `evaluate_model.py` to check train/test RMSE, MAE, and R2 for a feature set or
model choice without running the full prediction pipeline (no `--pred-gw`, no live
fixture/ownership network calls, no CSV outputs written).

Baseline run (default features, default model, single 70/30 split):

```powershell
python transfer/evaluate_model.py
```

Drop a feature to see its effect:

```powershell
python transfer/evaluate_model.py --exclude-features ict_index influence
```

Add a feature that isn't in the default set: first add it to `player_value_cols` in
both `get_ewma_df()` and `get_rolling_df()` in `wrangle_data_funcs.py` (so it gets
EWMA'd/rolled into the training data), then reference it:

```powershell
python transfer/evaluate_model.py --extra-features defensive_contribution
```

Swap model type:

```powershell
python transfer/evaluate_model.py --model xgboost
```

Check whether a change is a real improvement or noise, by rerunning the split with
multiple random seeds and comparing mean +/- std (useful especially for GK, which has
the fewest samples and the noisiest single-split RMSE):

```powershell
python transfer/evaluate_model.py --repeats 10
python transfer/evaluate_model.py --repeats 10 --exclude-features ict_index influence
```

See which features matter most (permutation importance — model-agnostic, so it's
comparable across `--model` choices):

```powershell
python transfer/evaluate_model.py --show-importance
```

Try the team/opponent strength ratings (FPL's official `strength_overall/attack/defence_home/away`
per team, venue-matched to each fixture — `team_strength_*` / `opponent_strength_*`).
These aren't in the default feature list yet; add them with `--extra-features` to
evaluate before promoting them into `get_features()` in `predict_gw_scores.py`. Note
`_attack`/`_defence` values come back as `0` for a season before FPL has computed them
(e.g. pre-season) — `_overall` is populated earlier and is a reasonable fallback:

```powershell
python transfer/evaluate_model.py --extra-features team_strength_attack team_strength_defence team_strength_overall opponent_strength_attack opponent_strength_defence opponent_strength_overall --model xgboost --show-importance
```

Log a run to CSV for side-by-side comparison later:

```powershell
python transfer/evaluate_model.py --model ridge --repeats 10 --save-csv transfer/outputs/eval_runs.csv
```

Full override of the feature list (bypasses `get_features()` entirely):

```powershell
python transfer/evaluate_model.py --features ewma_total_points value minutes
```

Try the home-advantage flag (also not in the default feature list yet):

```powershell
python transfer/evaluate_model.py --extra-features was_home --show-importance
```

## 4) Tuning Hyperparameters

Use `tune_model.py` to search each model family's hyperparameters against the same
train/test harness `evaluate_model.py` uses. Nothing is auto-promoted — review the
printed results and copy the winning config into `_DEFAULT_PARAMS` in
`modelling_funcs.py` yourself.

```powershell
python transfer/tune_model.py --model elasticnet
python transfer/tune_model.py --model all --repeats 3
```
