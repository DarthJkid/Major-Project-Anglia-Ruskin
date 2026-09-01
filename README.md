# Football Player Market Value Prediction

A machine-learning project that predicts the transfer-market value of professional
footballers and surfaces the results through an interactive **Streamlit dashboard**.
Final-year major project, Anglia Ruskin University.

> **Live app:** [Here](https://major-project-anglia-ruskin-ha2eyfpahrhxar5dc73twu.streamlit.app/)

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
Averaged across **73** logged experiment(s) (last updated 2026-09-01):

| Metric | All-time avg | Latest | Best |
| --- | --- | --- | --- |
| R² | 0.940 | 0.939 | 0.947 |
| RMSE | €2.69M | €2.70M | €2.49M |
| MAE | €1.00M | €1.00M | €1.00M |
| Accuracy @ 10% | 30.8% | 30.8% | 30.8% |
| Accuracy @ 20% | 55.9% | 55.9% | 55.9% |
| MAPE | 22.3% | 22.3% | 22.3% |
| Mean % error | +3.09% | +3.10% | +3.08% |
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
