# Football Player Market Value Prediction

A machine-learning project that predicts the transfer-market value of professional
footballers and surfaces the results through an interactive **Streamlit dashboard**.
Final-year major project, Anglia Ruskin University.

> **Live app:** _add your Streamlit Community Cloud URL here once deployed_

---

## What it does

The model estimates each player's market value (€) from in-game ability ratings
(EA Sports FC), real-world performance data (FBref) and historical transfer
valuations (Transfermarkt). The dashboard then lets you:

- **Player Lookup** — predicted vs. actual value, profile and a plain-English explanation of the valuation.
- **Deal Evaluator** — score a proposed transfer fee (0–100) against the model's prediction.
- **Market Inefficiencies** — ranked tables of the most under- and over-valued players, plus league-level aggregates.
- **Uncertainty** — how stable each prediction is across repeated model runs.
- **Insights Dashboard** — actual-vs-predicted calibration and global SHAP feature importance.
- **Model Metrics** — headline accuracy metrics and per-run results.

## Model performance

The table below is generated from the experiment history and refreshes whenever
a new run is logged (see [Experiment tracking](#experiment-tracking)).

<!-- EXPERIMENT_METRICS:START -->
Averaged across **1** logged experiment(s) (last updated 2026-06-22):

| Metric | All-time avg | Latest | Best |
| --- | --- | --- | --- |
| R² | 0.947 | 0.947 | 0.947 |
| RMSE | €2.49M | €2.49M | €2.49M |
| MAE | €1.00M | €1.00M | €1.00M |
| Accuracy @ 10% | 29.6% | 29.6% | 29.6% |
| Accuracy @ 20% | 54.5% | 54.5% | 54.5% |
| MAPE | 23.2% | 23.2% | 23.2% |
| Mean % error | +3.50% | +3.50% | +3.50% |
<!-- EXPERIMENT_METRICS:END -->

The final estimate is a hybrid (stacked) model that combines a gradient-boosted
learner with a neural network, weighted per run.

## Data sources

| Source | Used for |
| --- | --- |
| EA Sports FC (SoFIFA) | Overall rating, potential, wage, contract, position |
| FBref | Match performance: minutes, goals, assists, shots, per-90 metrics |
| Transfermarkt | Historical market valuations (target variable) and contract data |

## Repository structure

```
.
├── streamlit_app.py        # The dashboard (Streamlit Cloud entry point)
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml  # Theme + server config
├── data/                   # Runtime data the app loads
│   ├── final_player_values.csv
│   ├── metrics_summary.{json,csv}
│   ├── all_runs_metrics.csv
│   ├── shap_summary.csv
│   ├── player_stats_cleaned.csv
│   ├── transfermarkt_merged_players_with_valuation.csv
│   └── Fbref_Final_Data.csv
├── scripts/
│   ├── notebooks/          # Data cleaning + modelling notebooks
│   └── scrapers/           # FBref / SoFIFA / WhoScored scrapers
└── archive/                # Raw & intermediate data, not needed at runtime
```

## Run locally

Requires Python 3.10+.

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the dashboard
streamlit run streamlit_app.py
```

The app opens at <http://localhost:8501>.

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **Create app** → select this repo, branch, and `streamlit_app.py` as the entry point.
4. Deploy. The app installs `requirements.txt` automatically and reads everything from `data/`.

## Reproducing the model

The cleaning and modelling pipeline lives in `scripts/notebooks/`. Raw and
intermediate datasets are kept in `archive/` for reference; the cleaned inputs
the app needs are already in `data/`.

## Experiment tracking

Each model run's headline metrics are tracked over time so performance can be
compared across changes. After a run refreshes `data/metrics_summary.json`,
record it:

```bash
# log the latest run (optionally with a note) and refresh the README table
python scripts/experiment_tracker.py log --notes "added contract feature" --readme
```

This appends a row to `experiments/experiment_log.csv` (the running history),
archives the run's full output set to `experiments/runs/<id>/`, and regenerates
the [all-time-average table](#model-performance) above. Other commands:

```bash
python scripts/experiment_tracker.py promote-best    # display the best run's predictions
python scripts/experiment_tracker.py prune --keep 30 # trim old run archives
python scripts/experiment_tracker.py readme          # refresh README table only
python scripts/experiment_tracker.py show            # print the current summary
```

### Best run is what gets displayed

Every run is archived, but the dashboard always shows the **best** experiment
(highest R²), not just the latest. `promote-best` copies that run's outputs into
`data/`, so a worse run never replaces a better one.

### Automated daily runs

`.github/workflows/daily-model.yml` runs the pipeline on a schedule (03:00 UTC
daily, or manually from the Actions tab). It retrains, logs the experiment,
promotes the best run, refreshes the README, and commits the results back —
which triggers a Streamlit Cloud redeploy.

The orchestration lives in [`scripts/run_pipeline.py`](scripts/run_pipeline.py).
**Before the workflow can train**, port your modelling code from
`scripts/notebooks/` into that file's `train_model()` (it must write the output
files into `data/`). Training dependencies are in `requirements-train.txt`.
