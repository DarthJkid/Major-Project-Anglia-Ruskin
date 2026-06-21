"""Daily model pipeline entry point (used by .github/workflows/daily-model.yml).

What it does, in order:

  1. ``train_model()`` retrains the model and (re)writes the output files into
     the ``data/`` folder:
         final_player_values.csv, metrics_summary.json, metrics_summary.csv,
         all_runs_metrics.csv, shap_summary.csv
  2. The run is logged to the experiment history and its outputs are archived.
  3. The *best* experiment so far (highest R2) is promoted back into ``data/``
     so the dashboard always displays the best model, not just the latest.
  4. The README all-time-average table is refreshed.

You must implement ``train_model()`` with your modelling code, ported from
``scripts/notebooks/`` (CatBoost + LightGBM + XGBoost + MLPRegressor hybrid).
Until it is implemented this script fails fast, so the workflow never commits
stale or empty results.

Run locally:

    python scripts/run_pipeline.py --notes "manual run"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from experiment_tracker import log_experiment, promote_best, prune, update_readme


def train_model() -> None:
    """Retrain the model and write outputs into the data/ folder.

    Replace the body below with the modelling code from scripts/notebooks/.
    It must write at least data/final_player_values.csv and
    data/metrics_summary.json (plus the other DISPLAY_OUTPUTS where available).
    """
    raise NotImplementedError(
        "Port your training code from scripts/notebooks/ into train_model(). "
        "It must (re)write the model outputs into the data/ folder."
    )


def main(notes: str = "daily automated run", keep: int = 30) -> None:
    train_model()
    log_experiment(notes=notes)
    promote_best()
    prune(keep=keep)
    update_readme()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--notes", default="daily automated run")
    parser.add_argument("--keep", type=int, default=30, help="Run archives to retain.")
    args = parser.parse_args()
    main(notes=args.notes, keep=args.keep)
