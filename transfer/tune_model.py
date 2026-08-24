"""Search hyperparameters for a model family (or all of them) against the same
train/test harness evaluate_model.py uses.

Standalone: no --pred-gw, no network calls beyond training-data assembly. Nothing is
auto-promoted into modelling_funcs.py's _DEFAULT_PARAMS -- review the printed results
and copy the winning config in yourself.
"""

import argparse
import csv
import itertools
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

from evaluate_model import POSITIONS, aggregate, resolve_features
from modelling_funcs import create_model, summarize_metrics
from predict_gw_scores import get_model_func, get_training_df

# Small default search grid per model family. xgboost's grid is intentionally the
# largest (most hyperparameters worth tuning); --n-iter randomly subsamples it (and
# any other grid bigger than --n-iter) rather than always running the full product.
GRIDS = {
    'elasticnet': {'alpha': [0.01, 0.03, 0.1, 0.3, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
    'ridge': {'alpha': [0.1, 0.3, 1.0, 3.0, 10.0]},
    'lasso': {'alpha': [0.01, 0.03, 0.1, 0.3]},
    'linear': {},
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.03, 0.05, 0.1],
        'reg_lambda': [1, 5, 10],
    },
}
ALL_MODEL_NAMES = ['elasticnet', 'ridge', 'lasso', 'linear', 'xgboost']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search hyperparameters for a model family (or --model all) "
                     "using the same train/test harness as evaluate_model.py."
    )
    parser.add_argument("--training-years", nargs="+", type=int, default=[24, 25, 26], help="Training season suffixes, e.g. 24 25 26.")
    parser.add_argument("--training-n-gws", type=int, default=38, help="Number of GWs to include from each training year.")
    parser.add_argument("--alpha", type=float, default=0.6, help="EWMA alpha if using ewma averaging.")
    parser.add_argument("--rolling-gws", type=int, default=4, help="Rolling window size if using rolling averaging.")
    parser.add_argument("--avg-type", choices=["rolling", "ewma"], default="ewma", help="Feature averaging strategy.")
    parser.add_argument("--model", choices=ALL_MODEL_NAMES + ["all"], default="ridge", help="Model family to tune, or 'all' to sweep every family.")

    parser.add_argument("--features", nargs="+", default=None, help="Full override of the feature list (replaces get_features() entirely).")
    parser.add_argument("--exclude-features", nargs="+", default=None, help="Feature names to drop from the default get_features() list.")
    parser.add_argument("--extra-features", nargs="+", default=None, help="Additional column names to add to the default get_features() list (must already exist as columns in the built training dataframe).")

    parser.add_argument("--repeats", type=int, default=1, help="Random-seed repeats per hyperparameter combo (mean+-std), same as evaluate_model.py --repeats.")
    parser.add_argument("--n-iter", type=int, default=20, help="Max hyperparameter combos to try per model family; larger grids are randomly subsampled (fixed seed, reproducible).")
    parser.add_argument("--top-n", type=int, default=5, help="How many top combos to print per model family.")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional path to append one row per combo tried to a CSV log.")
    return parser.parse_args()


def generate_combos(grid, n_iter, seed=42):
    if not grid:
        return [{}]
    keys = list(grid.keys())
    all_combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    if len(all_combos) <= n_iter:
        return all_combos
    return random.Random(seed).sample(all_combos, n_iter)


def tune_one_model(training_df, features, model_name, args):
    model_func = get_model_func(model_name)
    combos = generate_combos(GRIDS[model_name], args.n_iter)
    results = []
    for combo in combos:
        per_pos_runs = {pos: [] for pos in POSITIONS}
        overall_runs = []
        for i in range(args.repeats):
            seed = 42 + i
            _, metrics_dict, _ = create_model(
                training_df, features, model_func, test=True, random_state=seed, model_params=combo
            )
            for pos, m in metrics_dict.items():
                if m is not None:
                    per_pos_runs[pos].append(m)
            overall = summarize_metrics(metrics_dict)
            if overall is not None:
                overall_runs.append(overall)
        results.append({'model': model_name, 'params': combo, 'per_pos_runs': per_pos_runs, 'overall_runs': overall_runs})
    return results


def mean_rmse(runs):
    return statistics.mean(r['rmse'] for r in runs)


def print_top_results(model_name, results, top_n):
    sortable = sorted((r for r in results if r['overall_runs']), key=lambda r: mean_rmse(r['overall_runs']))
    print(f"\n=== {model_name}: top {min(top_n, len(sortable))} of {len(results)} configs tried (ranked by OVERALL RMSE) ===")
    header = f"{'Params':<60}{'RMSE':>18}{'MAE':>18}{'R2':>18}"
    print(header)
    print("-" * len(header))
    for r in sortable[:top_n]:
        print(f"{str(r['params']):<60}{aggregate(r['overall_runs'], 'rmse'):>18}{aggregate(r['overall_runs'], 'mae'):>18}{aggregate(r['overall_runs'], 'r2'):>18}")


def print_per_position_best(model_name, results):
    print(f"\n{model_name}: best config per position (diagnostic only -- today's "
          f"architecture applies one shared config to every position; promoting a "
          f"per-position result would need extending create_model to accept a "
          f"{{position: params}} mapping, not done here):")
    for pos in POSITIONS:
        candidates = [r for r in results if r['per_pos_runs'][pos]]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: mean_rmse(r['per_pos_runs'][pos]))
        print(f"  {pos:<6} {str(best['params']):<55} RMSE={aggregate(best['per_pos_runs'][pos], 'rmse')}")


def print_global_best(all_results, top_n):
    sortable = sorted((r for r in all_results if r['overall_runs']), key=lambda r: mean_rmse(r['overall_runs']))
    print(f"\n=== Global best across all model families (top {min(top_n, len(sortable))}) ===")
    header = f"{'Model':<12}{'Params':<50}{'RMSE':>18}{'MAE':>18}{'R2':>18}"
    print(header)
    print("-" * len(header))
    for r in sortable[:top_n]:
        print(f"{r['model']:<12}{str(r['params']):<50}{aggregate(r['overall_runs'], 'rmse'):>18}{aggregate(r['overall_runs'], 'mae'):>18}{aggregate(r['overall_runs'], 'r2'):>18}")
    if sortable:
        best = sortable[0]
        print(f"\nReady to paste into modelling_funcs.py's _DEFAULT_PARAMS[{best['model']} class] "
              f"if you want to promote it: {best['params']}")


def append_csv_row(path, model_name, params, overall_runs):
    path = Path(path)
    is_new = not path.exists()
    row = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model': model_name,
        'params': str(params),
        'rmse_mean': round(statistics.mean(r['rmse'] for r in overall_runs), 3) if overall_runs else None,
        'rmse_std': round(statistics.stdev(r['rmse'] for r in overall_runs), 3) if len(overall_runs) > 1 else None,
        'mae_mean': round(statistics.mean(r['mae'] for r in overall_runs), 3) if overall_runs else None,
        'r2_mean': round(statistics.mean(r['r2'] for r in overall_runs), 3) if overall_runs else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    training_df = get_training_df(
        args.training_years, args.training_n_gws, args.avg_type, args.alpha, args.rolling_gws
    )
    features = resolve_features(args, training_df)

    model_names = ALL_MODEL_NAMES if args.model == "all" else [args.model]

    all_results = []
    for model_name in model_names:
        n_combos = len(generate_combos(GRIDS[model_name], args.n_iter))
        print(f"\n### Tuning {model_name} ({n_combos} configs x {args.repeats} repeat(s)) ###")
        results = tune_one_model(training_df, features, model_name, args)
        all_results.extend(results)
        print_top_results(model_name, results, args.top_n)
        print_per_position_best(model_name, results)
        if args.save_csv:
            for r in results:
                append_csv_row(args.save_csv, r['model'], r['params'], r['overall_runs'])

    if len(model_names) > 1:
        print_global_best(all_results, args.top_n)

    if args.save_csv:
        print(f"\nAppended {len(all_results)} run(s) to {args.save_csv}")


if __name__ == "__main__":
    main()
