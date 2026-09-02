"""Evaluate train/test performance for a feature set + model choice.

Standalone from predict_gw_scores.py: no --pred-gw, no live fixture/ownership network
calls, no prediction CSVs written. Just training-data assembly + the train/test split +
metrics, so you can quickly iterate on features/model type without running the full
weekly prediction pipeline.
"""

import argparse
import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path

from modelling_funcs import (
    compute_permutation_importance,
    create_model,
    summarize_importance,
    summarize_metrics,
)
from predict_gw_scores import get_features, get_model_func, get_training_df

POSITIONS = ['GK', 'DEF', 'MID', 'FWD']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate train/test RMSE/MAE/R2 (and optionally per-feature "
                     "importance) for a feature set + model choice, without running "
                     "the full prediction pipeline."
    )
    parser.add_argument("--training-years", nargs="+", type=int, default=[24, 25, 26], help="Training season suffixes, e.g. 24 25 26.")
    parser.add_argument("--training-n-gws", type=int, default=38, help="Number of GWs to include from each training year.")
    parser.add_argument("--alpha", type=float, default=0.6, help="EWMA alpha if using ewma averaging.")
    parser.add_argument("--rolling-gws", type=int, default=4, help="Rolling window size if using rolling averaging.")
    parser.add_argument("--avg-type", choices=["rolling", "ewma"], default="ewma", help="Feature averaging strategy.")
    parser.add_argument("--model", choices=["elasticnet", "ridge", "lasso", "linear", "xgboost"], default="elasticnet", help="Model family.")

    parser.add_argument("--features", nargs="+", default=None, help="Full override of the feature list (replaces get_features() entirely).")
    parser.add_argument("--exclude-features", nargs="+", default=None, help="Feature names to drop from the default get_features() list.")
    parser.add_argument("--extra-features", nargs="+", default=None, help="Additional column names to add to the default get_features() list (must already exist as columns in the built training dataframe).")

    parser.add_argument("--repeats", type=int, default=1, help="Number of train/test splits to run with different random seeds; reports mean+-std. Default 1 reproduces today's single fixed-seed (42) result.")
    parser.add_argument("--show-importance", action="store_true", help="Print permutation importance per feature/position (based on a single seed-42 split, regardless of --repeats).")
    parser.add_argument("--importance-repeats", type=int, default=10, help="Shuffle repeats per feature for permutation importance.")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional path to append one summary row (config + overall metrics) to a CSV log.")
    return parser.parse_args()


def resolve_features(args, training_df):
    if args.features and (args.exclude_features or args.extra_features):
        raise SystemExit("--features cannot be combined with --exclude-features/--extra-features; use one or the other.")

    available_cols = set(training_df.columns)

    if args.features is not None:
        requested = list(args.features)
    else:
        requested = get_features()
        if args.exclude_features:
            unknown = [f for f in args.exclude_features if f not in requested]
            if unknown:
                raise SystemExit(
                    f"--exclude-features contains names not in the default feature list "
                    f"(from get_features() in predict_gw_scores.py): {unknown}\n"
                    f"Default features are: {requested}"
                )
            requested = [f for f in requested if f not in args.exclude_features]
        if args.extra_features:
            requested = requested + [f for f in args.extra_features if f not in requested]

    missing = [f for f in requested if f not in available_cols]
    if missing:
        raise SystemExit(
            f"The following requested feature(s) are not columns in the built training "
            f"dataframe: {missing}\n"
            f"If these are raw per-GW stat columns (e.g. from the scraped gw CSVs) that "
            f"aren't yet averaged into the training data, add them to `player_value_cols` "
            f"in both get_ewma_df() and get_rolling_df() in wrangle_data_funcs.py first, "
            f"then rerun. Available columns in the current training dataframe:\n"
            f"{sorted(available_cols)}"
        )
    return requested


def run_eval(training_df, features, model_func, repeats):
    """Runs create_model(test=True) `repeats` times with distinct random seeds
    (42, 43, 44, ... so repeats=1 reproduces today's single fixed-seed-42 result
    exactly), collecting per-position and overall metrics. Also returns the fitted
    model_dict/metrics_dict from the first repeat (seed 42) for reuse by importance
    analysis, so that doesn't need a separate fit.
    """
    per_pos_runs = {pos: [] for pos in POSITIONS}
    overall_runs = []
    first_model_dict = None
    first_metrics_dict = None

    for i in range(repeats):
        seed = 42 + i
        model_dict, metrics_dict, _ = create_model(training_df, features, model_func, test=True, random_state=seed)
        if i == 0:
            first_model_dict, first_metrics_dict = model_dict, metrics_dict
        for pos, m in metrics_dict.items():
            if m is not None:
                per_pos_runs[pos].append(m)
        overall = summarize_metrics(metrics_dict)
        if overall is not None:
            overall_runs.append(overall)

    return per_pos_runs, overall_runs, first_model_dict, first_metrics_dict


def aggregate(runs, key):
    values = [r[key] for r in runs]
    if len(values) == 1:
        return f"{values[0]:.3f}"
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    return f"{mean:.3f} +/- {std:.3f}"


def print_report(per_pos_runs, overall_runs, args, features):
    print(f"\n=== Model evaluation: model={args.model} avg_type={args.avg_type} "
          f"repeats={args.repeats} n_features={len(features)} ===")
    print(f"Features: {features}\n")

    header = f"{'Position':<10}{'n_test':>8}{'RMSE':>18}{'MAE':>18}{'R2':>18}"
    print(header)
    print("-" * len(header))
    for pos in POSITIONS:
        runs = per_pos_runs[pos]
        if not runs:
            print(f"{pos:<10}{'(no data)':>8}")
            continue
        n_test = runs[0]['n_test']
        print(f"{pos:<10}{n_test:>8}{aggregate(runs, 'rmse'):>18}{aggregate(runs, 'mae'):>18}{aggregate(runs, 'r2'):>18}")
        if pos == 'GK' and n_test < 30:
            print(f"  (warning: GK test set is small (n={n_test}); metrics may be noisy - consider --repeats 10+)")

    print("-" * len(header))
    if overall_runs:
        n_total = overall_runs[0]['n_test']
        print(f"{'OVERALL':<10}{n_total:>8}{aggregate(overall_runs, 'rmse'):>18}"
              f"{aggregate(overall_runs, 'mae'):>18}{aggregate(overall_runs, 'r2'):>18}")
    print()


def print_importance(importance_dict, metrics_dict, features):
    overall = summarize_importance(importance_dict, metrics_dict)
    ordered_features = sorted(features, key=lambda f: overall.get(f, float('-inf')), reverse=True)

    print("=== Permutation importance (test RMSE increase when feature is shuffled; "
          "higher = more important; based on a single 70/30 split, seed 42) ===")
    header = f"{'Feature':<32}" + "".join(f"{pos:>10}" for pos in POSITIONS) + f"{'OVERALL':>10}"
    print(header)
    print("-" * len(header))
    for f in ordered_features:
        row = f"{f:<32}"
        for pos in POSITIONS:
            val = importance_dict.get(pos, {}).get(f)
            row += f"{val:>10.3f}" if val is not None else f"{'--':>10}"
        row += f"{overall.get(f, float('nan')):>10.3f}"
        print(row)
    print()


def append_csv_row(path, args, features, overall_runs):
    path = Path(path)
    is_new = not path.exists()
    row = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model': args.model,
        'avg_type': args.avg_type,
        'training_years': ' '.join(str(y) for y in args.training_years),
        'training_n_gws': args.training_n_gws,
        'n_features': len(features),
        'features': ' '.join(features),
        'repeats': args.repeats,
        'overall_rmse_mean': round(statistics.mean(r['rmse'] for r in overall_runs), 3) if overall_runs else None,
        'overall_rmse_std': round(statistics.stdev(r['rmse'] for r in overall_runs), 3) if len(overall_runs) > 1 else None,
        'overall_mae_mean': round(statistics.mean(r['mae'] for r in overall_runs), 3) if overall_runs else None,
        'overall_r2_mean': round(statistics.mean(r['r2'] for r in overall_runs), 3) if overall_runs else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended run summary to {path}")


def main():
    args = parse_args()
    training_df = get_training_df(
        args.training_years, args.training_n_gws, args.avg_type, args.alpha, args.rolling_gws
    )
    features = resolve_features(args, training_df)
    model_func = get_model_func(args.model)

    per_pos_runs, overall_runs, model_dict, metrics_dict = run_eval(training_df, features, model_func, args.repeats)
    print_report(per_pos_runs, overall_runs, args, features)

    if args.show_importance:
        importance_dict = compute_permutation_importance(
            model_dict, metrics_dict, features, n_repeats=args.importance_repeats
        )
        print_importance(importance_dict, metrics_dict, features)

    if args.save_csv:
        append_csv_row(args.save_csv, args, features, overall_runs)


if __name__ == "__main__":
    main()
