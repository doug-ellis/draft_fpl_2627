import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run weekly FPL prediction update for a target GW.")
    parser.add_argument("--pred-gw", type=int, required=True, help="Target gameweek to predict.")
    parser.add_argument("--pred-year", type=int, default=27, help="Prediction season suffix, e.g. 27 for 2026-27.")
    parser.add_argument("--model", choices=["elasticnet", "ridge", "lasso", "linear", "xgboost"], default="elasticnet")
    parser.add_argument("--skip-eval", action="store_true", help="Skip train/test RMSE printout.")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip FPL data scrape (use existing local data).")
    return parser.parse_args()


def run_scraper(transfer_dir):
    scraper = transfer_dir.parent.parent / "Fantasy-Premier-League" / "global_scraper.py"
    print("Scraping FPL data (this takes ~10-20 minutes)...")
    subprocess.run([sys.executable, str(scraper)], cwd=scraper.parent, check=True)


def run_predictions(args, transfer_dir):
    predict_script = transfer_dir / "predict_gw_scores.py"
    cmd = [
        sys.executable, str(predict_script),
        "--pred-gw", str(args.pred_gw),
        "--pred-year", str(args.pred_year),
        "--model", args.model,
    ]
    if args.skip_eval:
        cmd.append("--skip-eval")
    subprocess.run(cmd, cwd=transfer_dir, check=True)


def create_gw_notebook(pred_gw, transfer_dir):
    outputs_dir = transfer_dir / "outputs"
    template = outputs_dir / "looking_latest.ipynb"
    dest = outputs_dir / f"looking_gw{pred_gw}.ipynb"
    if dest.exists():
        print(f"Notebook {dest.name} already exists, skipping creation.")
        return
    content = template.read_text(encoding="utf-8")
    content, n_subs = re.subn(r"TARGET_GW = None", f"TARGET_GW = {pred_gw}", content)
    if n_subs == 0:
        print(f"Warning: TARGET_GW placeholder not found in {template.name}; notebook created unmodified.")
    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest.name}")


def main():
    args = parse_args()
    transfer_dir = Path(__file__).resolve().parent

    if not args.skip_scrape:
        run_scraper(transfer_dir)

    run_predictions(args, transfer_dir)
    create_gw_notebook(args.pred_gw, transfer_dir)

    print(f"\nDone. GW {args.pred_gw} predictions ready.")
    print(f"Open transfer/outputs/looking_gw{args.pred_gw}.ipynb to analyse.")


if __name__ == "__main__":
    main()
