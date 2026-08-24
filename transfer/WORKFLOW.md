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
