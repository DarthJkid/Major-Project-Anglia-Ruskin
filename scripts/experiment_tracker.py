"""Track model experiments, tests and metrics over time.

Every time the modelling pipeline runs and refreshes ``data/metrics_summary.json``
(and ``data/all_runs_metrics.csv``), record that run here so performance can be
compared across days/changes:

    # from the repo root, after a model run
    python scripts/experiment_tracker.py log --notes "added contract feature"

    # refresh the all-time-average table in README.md
    python scripts/experiment_tracker.py readme

    # do both in one go
    python scripts/experiment_tracker.py log --notes "..." --readme

Or call it from a notebook at the end of training:

    import sys; sys.path.append("scripts")
    from experiment_tracker import log_experiment, update_readme
    log_experiment(notes="baseline")
    update_readme()

Each call appends one row to ``experiments/experiment_log.csv`` (the running
history) and writes a full snapshot to ``experiments/runs/<id>.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "experiments"
RUNS_DIR = EXP_DIR / "runs"
LOG_FILE = EXP_DIR / "experiment_log.csv"
README_FILE = ROOT / "README.md"

DEFAULT_METRICS_JSON = DATA_DIR / "metrics_summary.json"
DEFAULT_ALL_RUNS_CSV = DATA_DIR / "all_runs_metrics.csv"

# Headline metrics recorded per experiment, with the direction that counts as
# "better" (used to pick the all-time best). "zero" means closest to 0 is best.
METRIC_DIRECTION = {
    "R2": "higher",
    "RMSE": "lower",
    "MAE": "lower",
    "Accuracy@10%": "higher",
    "Accuracy@20%": "higher",
    "MAPE": "lower",
    "Mean Percentage Error": "zero",
}
TRACKED_METRICS = list(METRIC_DIRECTION)

README_START = "<!-- EXPERIMENT_METRICS:START -->"
README_END = "<!-- EXPERIMENT_METRICS:END -->"

# Model output files archived per run and promoted to data/ for the live app.
DISPLAY_OUTPUTS = [
    "final_player_values.csv",
    "metrics_summary.json",
    "metrics_summary.csv",
    "all_runs_metrics.csv",
    "shap_summary.csv",
]

# Metric used to decide which experiment's predictions are displayed.
PROMOTE_METRIC = "R2"


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
def _read_metric_means(metrics_json: Path) -> dict:
    """Return {metric: mean} from a metrics_summary.json file."""
    data = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    means = {}
    for key, val in data.items():
        if isinstance(val, dict) and "mean" in val:
            means[key] = val["mean"]
        elif isinstance(val, (int, float)):
            means[key] = val
    return means


def log_experiment(
    metrics_json: Path = DEFAULT_METRICS_JSON,
    all_runs_csv: Path = DEFAULT_ALL_RUNS_CSV,
    notes: str = "",
) -> dict:
    """Append the current model run's metrics to the experiment history.

    Returns the recorded row as a dict.
    """
    metrics_json = Path(metrics_json)
    if not metrics_json.exists():
        raise FileNotFoundError(
            f"{metrics_json} not found - run the modelling pipeline first."
        )

    EXP_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    means = _read_metric_means(metrics_json)

    timestamp = dt.datetime.now().replace(microsecond=0)
    experiment_id = timestamp.strftime("%Y%m%d_%H%M%S")

    n_runs = None
    all_runs_csv = Path(all_runs_csv)
    if all_runs_csv.exists():
        try:
            n_runs = int(len(pd.read_csv(all_runs_csv)))
        except Exception:
            n_runs = None

    record = {
        "experiment_id": experiment_id,
        "timestamp": timestamp.isoformat(),
        "notes": notes,
        "n_runs": n_runs,
    }
    for metric in TRACKED_METRICS:
        record[metric] = means.get(metric)

    # Archive this run's full output set so the best run can be promoted later.
    run_dir = RUNS_DIR / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    for name in DISPLAY_OUTPUTS:
        src = DATA_DIR / name
        if src.exists():
            shutil.copy2(src, run_dir / name)

    # Append to the running CSV history.
    new_row = pd.DataFrame([record])
    if LOG_FILE.exists():
        history = pd.read_csv(LOG_FILE)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row
    history.to_csv(LOG_FILE, index=False)

    print(f"Logged experiment {experiment_id} ({len(history)} total).")
    return record


def load_history() -> pd.DataFrame:
    """Load the full experiment history (empty DataFrame if none yet)."""
    if not LOG_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(LOG_FILE)


# ------------------------------------------------------------------
# All-time summary
# ------------------------------------------------------------------
def all_time_summary() -> dict:
    """Compute all-time average / latest / best for each tracked metric."""
    history = load_history()
    if history.empty:
        return {}

    latest = history.iloc[-1]
    summary = {"count": len(history), "metrics": {}}
    for metric, direction in METRIC_DIRECTION.items():
        if metric not in history.columns:
            continue
        series = pd.to_numeric(history[metric], errors="coerce").dropna()
        if series.empty:
            continue
        if direction == "higher":
            best = series.max()
        elif direction == "lower":
            best = series.min()
        else:  # closest to zero
            best = series.iloc[series.abs().argmin()]
        summary["metrics"][metric] = {
            "average": float(series.mean()),
            "latest": float(latest[metric]) if pd.notna(latest.get(metric)) else None,
            "best": float(best),
        }
    return summary


# ------------------------------------------------------------------
# Promotion: display the best experiment's predictions
# ------------------------------------------------------------------
def best_experiment(metric: str = PROMOTE_METRIC) -> dict | None:
    """Return the best experiment row (as a dict) by the given metric."""
    history = load_history()
    if history.empty or metric not in history.columns:
        return None
    values = pd.to_numeric(history[metric], errors="coerce")
    history = history.assign(_score=values).dropna(subset=["_score"])
    if history.empty:
        return None
    direction = METRIC_DIRECTION.get(metric, "higher")
    if direction == "higher":
        idx = history["_score"].idxmax()
    elif direction == "lower":
        idx = history["_score"].idxmin()
    else:  # closest to zero
        idx = history["_score"].abs().idxmin()
    return history.loc[idx].drop(labels="_score").to_dict()


def promote_best(metric: str = PROMOTE_METRIC) -> dict | None:
    """Copy the best experiment's archived outputs into data/ (displayed set)."""
    best = best_experiment(metric)
    if best is None:
        print("No experiments to promote yet.")
        return None

    run_dir = RUNS_DIR / str(best["experiment_id"])
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Archived outputs for best experiment {best['experiment_id']} "
            f"not found at {run_dir}."
        )

    promoted = []
    for name in DISPLAY_OUTPUTS:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, DATA_DIR / name)
            promoted.append(name)

    print(
        f"Promoted experiment {best['experiment_id']} "
        f"({metric}={best.get(metric)}): {len(promoted)} file(s) -> data/."
    )
    return best


def prune(keep: int = 30) -> None:
    """Delete old run archives, always keeping the best run and the latest N."""
    history = load_history()
    if history.empty:
        return
    keep_ids = set(history["experiment_id"].astype(str).tail(keep))
    best = best_experiment()
    if best is not None:
        keep_ids.add(str(best["experiment_id"]))

    removed = 0
    for child in RUNS_DIR.glob("*"):
        if child.is_dir() and child.name not in keep_ids:
            shutil.rmtree(child)
            removed += 1
    if removed:
        print(f"Pruned {removed} old run archive(s); kept {len(keep_ids)}.")


# ------------------------------------------------------------------
# Formatting + README
# ------------------------------------------------------------------
def _format_metric(metric: str, value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if metric == "R2":
        return f"{value:.3f}"
    if metric in ("RMSE", "MAE"):
        return f"€{value / 1e6:.2f}M"
    if metric in ("Accuracy@10%", "Accuracy@20%", "MAPE"):
        return f"{value:.1f}%"
    if metric == "Mean Percentage Error":
        return f"{value:+.2f}%"
    return f"{value:.3f}"


_METRIC_LABELS = {
    "R2": "R²",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "Accuracy@10%": "Accuracy @ 10%",
    "Accuracy@20%": "Accuracy @ 20%",
    "MAPE": "MAPE",
    "Mean Percentage Error": "Mean % error",
}


def render_markdown(summary: dict) -> str:
    """Build the README block for the all-time-average table."""
    if not summary:
        return (
            "_No experiments logged yet. Run "
            "`python scripts/experiment_tracker.py log` after a model run._"
        )

    today = dt.date.today().isoformat()
    lines = [
        f"Averaged across **{summary['count']}** logged experiment(s) "
        f"(last updated {today}):",
        "",
        "| Metric | All-time avg | Latest | Best |",
        "| --- | --- | --- | --- |",
    ]
    for metric in TRACKED_METRICS:
        m = summary["metrics"].get(metric)
        if not m:
            continue
        lines.append(
            f"| {_METRIC_LABELS[metric]} "
            f"| {_format_metric(metric, m['average'])} "
            f"| {_format_metric(metric, m['latest'])} "
            f"| {_format_metric(metric, m['best'])} |"
        )
    return "\n".join(lines)


def update_readme(readme: Path = README_FILE) -> None:
    """Rewrite the auto-generated metrics block between the README markers."""
    readme = Path(readme)
    block = render_markdown(all_time_summary())
    wrapped = f"{README_START}\n{block}\n{README_END}"

    text = readme.read_text(encoding="utf-8")
    if README_START in text and README_END in text:
        before = text.split(README_START)[0]
        after = text.split(README_END)[1]
        text = before + wrapped + after
    else:
        raise RuntimeError(
            f"README markers not found. Add the following where you want the "
            f"table:\n{README_START}\n{README_END}"
        )
    readme.write_text(text, encoding="utf-8")
    print(f"Updated {readme.name} all-time metrics block.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_log = sub.add_parser("log", help="Record the current model run's metrics.")
    p_log.add_argument("--notes", default="", help="Short note describing this run.")
    p_log.add_argument("--metrics", default=str(DEFAULT_METRICS_JSON))
    p_log.add_argument("--all-runs", default=str(DEFAULT_ALL_RUNS_CSV))
    p_log.add_argument("--readme", action="store_true", help="Also refresh README.")

    p_best = sub.add_parser(
        "promote-best", help="Copy the best experiment's outputs into data/."
    )
    p_best.add_argument("--metric", default=PROMOTE_METRIC)

    p_prune = sub.add_parser("prune", help="Remove old run archives.")
    p_prune.add_argument("--keep", type=int, default=30)

    sub.add_parser("readme", help="Refresh the all-time-average table in README.md.")
    sub.add_parser("show", help="Print the current all-time summary.")

    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "log":
        log_experiment(metrics_json=args.metrics, all_runs_csv=args.all_runs, notes=args.notes)
        if args.readme:
            update_readme()
    elif args.command == "promote-best":
        promote_best(metric=args.metric)
    elif args.command == "prune":
        prune(keep=args.keep)
    elif args.command == "readme":
        update_readme()
    elif args.command == "show":
        print(json.dumps(all_time_summary(), indent=2))


if __name__ == "__main__":
    main()
