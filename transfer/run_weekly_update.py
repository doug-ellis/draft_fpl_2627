import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run weekly FPL prediction update for a target GW.")
    parser.add_argument("--pred-gw", type=int, required=True, help="Target gameweek to predict.")
    parser.add_argument("--end-gw", type=int, default=None, help="Last gameweek to forecast (inclusive). Default: auto-detected end of season.")
    parser.add_argument("--pred-year", type=int, default=27, help="Prediction season suffix, e.g. 27 for 2026-27.")
    parser.add_argument("--model", choices=["elasticnet", "ridge", "lasso", "linear", "xgboost"], default="ridge")
    parser.add_argument("--skip-eval", action="store_true", help="Skip train/test RMSE printout.")
    parser.add_argument(
        "--skip-scrape", "--skip-scraping",
        dest="skip_scrape", action="store_true",
        help="Skip the FPL data scrape and use the most recent data already in transfer/outputs/scraped_data.",
    )
    return parser.parse_args()


def run_scraper(transfer_dir):
    scraper = transfer_dir.parent.parent / "Fantasy-Premier-League" / "global_scraper.py"
    scraped_data_dir = transfer_dir / "outputs" / "scraped_data"
    scraped_data_dir.mkdir(parents=True, exist_ok=True)
    print("Scraping FPL data (this takes ~10-20 minutes)...")
    # global_scraper.py writes to a `data/<season>/` path relative to cwd, so running it
    # with cwd set here lands the output at transfer/outputs/scraped_data/data/<season>/
    # instead of inside the separate Fantasy-Premier-League checkout.
    subprocess.run([sys.executable, str(scraper)], cwd=scraped_data_dir, check=True)


def run_predictions(args, transfer_dir):
    predict_script = transfer_dir / "predict_gw_scores.py"
    cmd = [
        sys.executable, str(predict_script),
        "--pred-gw", str(args.pred_gw),
        "--pred-year", str(args.pred_year),
        "--model", args.model,
    ]
    if args.end_gw is not None:
        cmd += ["--end-gw", str(args.end_gw)]
    if args.skip_eval:
        cmd.append("--skip-eval")
    subprocess.run(cmd, cwd=transfer_dir, check=True)


def create_gw_notebook(pred_gw, transfer_dir):
    outputs_dir = transfer_dir / "outputs"
    notebooks_dir = outputs_dir / "picking_notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    template = outputs_dir / "looking_latest.ipynb"
    dest = notebooks_dir / f"gw{pred_gw}.ipynb"
    if dest.exists():
        print(f"Notebook {dest.name} already exists, skipping creation.")
        return
    # The template is already authored for picking_notebooks/'s depth (one level
    # below outputs/): sys.path.append('../..'), predictions_dir/fixture_dir='../predictions'
    # etc. Only the TARGET_GW placeholder needs substituting per GW.
    content = template.read_text(encoding="utf-8")
    content, n_subs = re.subn(r"TARGET_GW = None", f"TARGET_GW = {pred_gw}", content)
    if n_subs == 0:
        print(f"Warning: TARGET_GW placeholder not found in {template.name}; notebook created unmodified.")

    dest.write_text(content, encoding="utf-8")
    print(f"Created {dest.relative_to(transfer_dir)}")


def main():
    args = parse_args()
    transfer_dir = Path(__file__).resolve().parent

    if not args.skip_scrape:
        run_scraper(transfer_dir)

    run_predictions(args, transfer_dir)
    create_gw_notebook(args.pred_gw, transfer_dir)

    horizon_note = f"GW {args.pred_gw} through end of season" if args.end_gw is None else f"GW {args.pred_gw} through {args.end_gw}"
    print(f"\nDone. {horizon_note} predictions ready.")
    print(f"Open transfer/outputs/picking_notebooks/gw{args.pred_gw}.ipynb to analyse.")


if __name__ == "__main__":
    main()
