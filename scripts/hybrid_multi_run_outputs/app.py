import os
import json
import re
import unicodedata
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# PATH SETUP
# ============================================================

APP_FILE = Path(__file__).resolve()

PROJECT_ROOT = APP_FILE.parent.parent.parent

# Your stable setup
DATA_DIR = APP_FILE.parent

st.set_page_config(page_title="Football Player Valuation App", layout="wide")

FINAL_VALUES_FILE = DATA_DIR / "final_player_values.csv"
MERGED_FILE = DATA_DIR / "merged_inference_dataset.csv"
METRICS_JSON_FILE = DATA_DIR / "metrics_summary.json"
METRICS_CSV_FILE = DATA_DIR / "metrics_summary.csv"
ALL_RUNS_METRICS_FILE = DATA_DIR / "all_runs_metrics.csv"
SHAP_FILE = DATA_DIR / "shap_summary.csv"

# Main raw datasets for metadata enrichment
EA_FILE = PROJECT_ROOT / "player_stats_cleaned.csv"
TM_FILE = PROJECT_ROOT / "transfermarkt_merged_players_with_valuation.csv"
FB_FILE = PROJECT_ROOT / "Fbref_Final_Data.csv"


# ============================================================
# HELPERS
# ============================================================

def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_name(name):
    if pd.isna(name):
        return np.nan
    name = str(name)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def format_eur(x):
    if pd.isna(x):
        return "N/A"
    return f"€{x:,.0f}"


def get_age_value(row):
    for c in ["age_at_valuation", "age_ea", "age"]:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return np.nan


def safe_column_subset(df, cols):
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def ensure_required_columns(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    if "player_name" not in df.columns:
        for c in ["full_name", "name"]:
            if c in df.columns:
                df["player_name"] = df[c]
                break
    if "player_name" not in df.columns:
        if "player_id" in df.columns:
            df["player_name"] = df["player_id"]
        else:
            df["player_name"] = "Unknown Player"

    if "name_key" not in df.columns:
        df["name_key"] = df["player_name"].apply(normalize_name)

    if "position_group" not in df.columns:
        df["position_group"] = "UNK"

    if "predicted_value_mean" not in df.columns:
        if "predicted_value" in df.columns:
            df["predicted_value_mean"] = df["predicted_value"]
        elif "predicted_value_median" in df.columns:
            df["predicted_value_mean"] = df["predicted_value_median"]
        else:
            run_cols = [c for c in df.columns if c.startswith("pred_run_")]
            if len(run_cols) > 0:
                df["predicted_value_mean"] = df[run_cols].mean(axis=1)

    if "predicted_value_std" not in df.columns:
        run_cols = [c for c in df.columns if c.startswith("pred_run_")]
        if len(run_cols) > 1:
            df["predicted_value_std"] = df[run_cols].std(axis=1)

    if "predicted_minus_actual" not in df.columns:
        if "predicted_value_mean" in df.columns and "actual_value" in df.columns:
            df["predicted_minus_actual"] = df["predicted_value_mean"] - df["actual_value"]

    if "percentage_diff" not in df.columns:
        if "predicted_value_mean" in df.columns and "actual_value" in df.columns:
            df["percentage_diff"] = (
                (df["predicted_value_mean"] - df["actual_value"])
                / df["actual_value"].replace(0, np.nan)
            ) * 100

    if "valuation_label" not in df.columns and "percentage_diff" in df.columns:
        df["valuation_label"] = np.where(df["percentage_diff"] > 0, "Undervalued", "Overvalued")

    return df


def build_simple_explanation(row):
    explanations = []

    if pd.notna(row.get("overall_rating")) and row.get("overall_rating", 0) >= 80:
        explanations.append("High overall rating is likely pushing the valuation upward.")

    if pd.notna(row.get("potential")) and row.get("potential", 0) >= 85:
        explanations.append("Strong potential suggests future upside and increases long-term value.")

    if pd.notna(row.get("goal_contrib_per90")) and row.get("goal_contrib_per90", 0) >= 0.50:
        explanations.append("Strong goal contribution per 90 indicates efficient attacking output.")

    if pd.notna(row.get("contract_years_left")) and row.get("contract_years_left", 0) >= 2:
        explanations.append("Longer contract length strengthens club leverage in the market.")

    if pd.notna(row.get("contract_years_left_ea")) and row.get("contract_years_left_ea", 0) >= 2:
        explanations.append("Longer contract length strengthens club leverage in the market.")

    age_value = get_age_value(row)
    if pd.notna(age_value):
        if 23 <= age_value <= 28:
            explanations.append("The player is near the typical peak-value age range.")
        elif age_value < 21:
            explanations.append("Youth may raise long-term upside, although uncertainty can be higher.")
        elif age_value > 30:
            explanations.append("Older age may reduce long-term resale value.")

    if pd.notna(row.get("Min")) and row.get("Min", 0) >= 1800:
        explanations.append("High minutes played suggest sustained involvement and stronger evidence of ability.")

    if pd.notna(row.get("predicted_value_std")):
        if row.get("predicted_value_std", 0) < 500000:
            explanations.append("Prediction uncertainty is relatively low across repeated runs.")
        else:
            explanations.append("Prediction uncertainty is relatively high across repeated runs, so this estimate should be treated more cautiously.")

    if not explanations:
        explanations.append("This valuation appears to reflect a combined effect of ability, performance, age, and market context.")

    return explanations


# ============================================================
# METADATA PREP
# ============================================================

def prepare_ea_metadata(ea):
    if ea is None or ea.empty:
        return None

    ea = ea.copy()

    if "full_name" in ea.columns:
        ea["player_name"] = ea["full_name"].fillna(ea.get("name"))
    elif "name" in ea.columns:
        ea["player_name"] = ea["name"]
    else:
        return None

    ea["name_key"] = ea["player_name"].apply(normalize_name)

    if "dob" in ea.columns:
        ea["dob"] = pd.to_datetime(ea["dob"], errors="coerce")
        ref_date = pd.Timestamp(datetime.now().date())
        ea["age_ea"] = (ref_date - ea["dob"]).dt.days / 365.25

    if "club_contract_valid_until" in ea.columns:
        ea["club_contract_valid_until"] = pd.to_datetime(ea["club_contract_valid_until"], errors="coerce")
        ref_date = pd.Timestamp(datetime.now().date())
        ea["contract_years_left_ea"] = ((ea["club_contract_valid_until"] - ref_date).dt.days / 365.25).clip(lower=0)

    keep_cols = [
        "name_key", "player_name", "overall_rating", "potential", "wage",
        "club_league_name", "age_ea", "contract_years_left_ea", "positions"
    ]
    keep_cols = [c for c in keep_cols if c in ea.columns]

    ea_meta = ea[keep_cols].copy()
    ea_meta = ea_meta.drop_duplicates("name_key")
    return ea_meta


def prepare_tm_metadata(tm):
    if tm is None or tm.empty:
        return None

    tm = tm.copy()

    if "name" not in tm.columns:
        return None

    tm["player_name"] = tm["name"]
    tm["name_key"] = tm["player_name"].apply(normalize_name)

    if "date" in tm.columns:
        tm["date"] = pd.to_datetime(tm["date"], errors="coerce")
        tm = tm.sort_values(["name_key", "date"], ascending=[True, False])

    tm = tm.drop_duplicates("name_key", keep="first")

    keep_cols = [
        "name_key", "player_name", "current_club_name_valuation",
        "player_club_domestic_competition_id", "age", "age_at_valuation",
        "contract_years_left", "contract_expiration_date", "sub_position"
    ]
    keep_cols = [c for c in keep_cols if c in tm.columns]

    tm_meta = tm[keep_cols].copy()

    if "current_club_name_valuation" in tm_meta.columns:
        tm_meta["club_name"] = tm_meta["current_club_name_valuation"]

    return tm_meta


def prepare_fb_metadata(fb):
    if fb is None or fb.empty:
        return None

    fb = fb.copy()

    if "Player" not in fb.columns:
        return None

    fb["player_name"] = fb["Player"]
    fb["name_key"] = fb["player_name"].apply(normalize_name)

    if {"name_key", "Min"}.issubset(fb.columns):
        fb["Min"] = pd.to_numeric(fb["Min"], errors="coerce")
        fb = fb.sort_values(["name_key", "Min"], ascending=[True, False]).drop_duplicates("name_key", keep="first")
    else:
        fb = fb.drop_duplicates("name_key", keep="first")

    if {"Gls", "Ast", "90s"}.issubset(fb.columns):
        fb["Gls"] = pd.to_numeric(fb["Gls"], errors="coerce")
        fb["Ast"] = pd.to_numeric(fb["Ast"], errors="coerce")
        fb["90s"] = pd.to_numeric(fb["90s"], errors="coerce")
        fb["goal_contrib_per90"] = np.where(fb["90s"] > 0, (fb["Gls"] + fb["Ast"]) / fb["90s"], np.nan)

    keep_cols = [
        "name_key", "player_name", "Comp", "Pos", "Age", "Min",
        "90s", "goal_contrib_per90", "Gls/90", "Sh/90", "SoT/90", "Int/90"
    ]
    keep_cols = [c for c in keep_cols if c in fb.columns]

    fb_meta = fb[keep_cols].copy()
    return fb_meta


def build_master_metadata(ea_meta, tm_meta, fb_meta):
    """
    Combine EA, TM, and FB metadata safely without repeated suffix collisions.
    """
    frames = [x for x in [ea_meta, tm_meta, fb_meta] if x is not None and not x.empty]

    if not frames:
        return None

    # Start from first frame
    master = frames[0].copy()

    # Merge the rest one by one, but rename overlapping columns first
    for i, df in enumerate(frames[1:], start=2):
        df = df.copy()

        overlap = [c for c in df.columns if c in master.columns and c != "name_key"]
        rename_map = {c: f"{c}_src{i}" for c in overlap}
        df = df.rename(columns=rename_map)

        master = master.merge(df, on="name_key", how="outer")

    # --------------------------------------------------------
    # Coalesce player_name
    # --------------------------------------------------------
    name_cols = [c for c in master.columns if c.startswith("player_name")]
    if "player_name" not in master.columns and name_cols:
        master["player_name"] = master[name_cols[0]]

    if "player_name" in master.columns:
        for c in name_cols:
            if c != "player_name":
                master["player_name"] = master["player_name"].fillna(master[c])

    # --------------------------------------------------------
    # Coalesce club_name
    # --------------------------------------------------------
    club_name_candidates = [c for c in master.columns if c in ["club_name", "current_club_name_valuation"] or c.startswith("club_name_") or c.startswith("current_club_name_valuation")]
    if "club_name" not in master.columns and club_name_candidates:
        master["club_name"] = master[club_name_candidates[0]]

    if "club_name" in master.columns:
        for c in club_name_candidates:
            if c != "club_name":
                master["club_name"] = master["club_name"].fillna(master[c])

    # --------------------------------------------------------
    # Coalesce league_name
    # --------------------------------------------------------
    league_candidates = [c for c in master.columns if c in ["league_name", "club_league_name", "Comp", "player_club_domestic_competition_id"] or c.startswith("league_name_") or c.startswith("club_league_name") or c.startswith("Comp_") or c.startswith("player_club_domestic_competition_id")]
    if "league_name" not in master.columns and league_candidates:
        master["league_name"] = master[league_candidates[0]]

    if "league_name" in master.columns:
        for c in league_candidates:
            if c != "league_name":
                master["league_name"] = master["league_name"].fillna(master[c])

    # --------------------------------------------------------
    # Coalesce position_group
    # --------------------------------------------------------
    position_candidates = [c for c in master.columns if c in ["position_group", "positions", "sub_position", "Pos"] or c.startswith("position_group_") or c.startswith("positions_") or c.startswith("sub_position_") or c.startswith("Pos_")]
    if "position_group" not in master.columns and position_candidates:
        master["position_group"] = master[position_candidates[0]]

    if "position_group" in master.columns:
        for c in position_candidates:
            if c != "position_group":
                master["position_group"] = master["position_group"].fillna(master[c])

    # --------------------------------------------------------
    # Coalesce useful numeric / context fields
    # --------------------------------------------------------
    coalesce_targets = [
        "overall_rating",
        "potential",
        "wage",
        "age_ea",
        "age",
        "age_at_valuation",
        "contract_years_left",
        "contract_years_left_ea",
        "goal_contrib_per90",
        "Min"
    ]

    for target in coalesce_targets:
        candidates = [c for c in master.columns if c == target or c.startswith(f"{target}_")]
        if candidates:
            if target not in master.columns:
                master[target] = master[candidates[0]]
            for c in candidates:
                if c != target:
                    master[target] = master[target].fillna(master[c])

    master = master.drop_duplicates("name_key")
    return master


def enrich_predictions(final_df, merged_df, master_meta):
    if final_df is None or final_df.empty:
        return final_df

    final_df = ensure_required_columns(final_df.copy())

    if "name_key" not in final_df.columns:
        final_df["name_key"] = final_df["player_name"].apply(normalize_name)

    # First enrich from merged file if available
    if merged_df is not None and not merged_df.empty:
        merged_df = merged_df.copy()
        if "player_name" not in merged_df.columns:
            for c in ["full_name", "name"]:
                if c in merged_df.columns:
                    merged_df["player_name"] = merged_df[c]
                    break
        if "player_name" not in merged_df.columns and "name_key" in merged_df.columns:
            merged_df["player_name"] = merged_df["name_key"]

        if "name_key" not in merged_df.columns and "player_name" in merged_df.columns:
            merged_df["name_key"] = merged_df["player_name"].apply(normalize_name)

        keep_cols = [
            "name_key", "player_name", "position_group", "club_league_name",
            "overall_rating", "potential", "wage",
            "age_ea", "age", "age_at_valuation",
            "contract_years_left", "contract_years_left_ea",
            "goal_contrib_per90", "Min"
        ]
        keep_cols = [c for c in keep_cols if c in merged_df.columns]
        merged_small = merged_df[keep_cols].drop_duplicates("name_key")

        final_df = final_df.merge(
            merged_small,
            on="name_key",
            how="left",
            suffixes=("", "_merged")
        )

    # Then enrich from master metadata
    if master_meta is not None and not master_meta.empty:
        meta_keep = [
            "name_key", "player_name", "position_group", "league_name", "club_name",
            "overall_rating", "potential", "wage", "age_ea", "age",
            "age_at_valuation", "contract_years_left", "contract_years_left_ea",
            "goal_contrib_per90", "Min"
        ]
        meta_keep = [c for c in meta_keep if c in master_meta.columns]
        master_small = master_meta[meta_keep].drop_duplicates("name_key")

        final_df = final_df.merge(
            master_small,
            on="name_key",
            how="left",
            suffixes=("", "_meta")
        )

    # fill player name
    for col in ["player_name_merged", "player_name_meta"]:
        if col in final_df.columns:
            final_df["player_name"] = final_df["player_name"].fillna(final_df[col])

    # fill position
    for col in ["position_group_merged", "position_group_meta"]:
        if col in final_df.columns:
            final_df["position_group"] = final_df["position_group"].replace("UNK", np.nan)
            final_df["position_group"] = final_df["position_group"].fillna(final_df[col])

    final_df["position_group"] = final_df["position_group"].fillna("UNK")

    # league / club fields
    if "club_league_name" not in final_df.columns:
        final_df["club_league_name"] = np.nan

    if "league_name" in final_df.columns:
        final_df["club_league_name"] = final_df["club_league_name"].fillna(final_df["league_name"])

    if "club_name" not in final_df.columns:
        final_df["club_name"] = np.nan

    # fill remaining useful fields
    fill_cols = [
        "overall_rating", "potential", "wage", "age_ea", "age",
        "age_at_valuation", "contract_years_left", "contract_years_left_ea",
        "goal_contrib_per90", "Min"
    ]

    for base_col in fill_cols:
        merged_col = f"{base_col}_merged"
        meta_col = f"{base_col}_meta"

        if merged_col in final_df.columns:
            if base_col not in final_df.columns:
                final_df[base_col] = np.nan
            final_df[base_col] = final_df[base_col].fillna(final_df[merged_col])

        if meta_col in final_df.columns:
            if base_col not in final_df.columns:
                final_df[base_col] = np.nan
            final_df[base_col] = final_df[base_col].fillna(final_df[meta_col])

    final_df = ensure_required_columns(final_df)

    return final_df


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    final_df = safe_read_csv(FINAL_VALUES_FILE)
    merged_df = safe_read_csv(MERGED_FILE)
    metrics_json = safe_read_json(METRICS_JSON_FILE)
    metrics_csv = safe_read_csv(METRICS_CSV_FILE)
    all_runs_metrics = safe_read_csv(ALL_RUNS_METRICS_FILE)
    shap_df = safe_read_csv(SHAP_FILE)

    ea_raw = safe_read_csv(EA_FILE)
    tm_raw = safe_read_csv(TM_FILE)
    fb_raw = safe_read_csv(FB_FILE)

    ea_meta = prepare_ea_metadata(ea_raw)
    tm_meta = prepare_tm_metadata(tm_raw)
    fb_meta = prepare_fb_metadata(fb_raw)
    master_meta = build_master_metadata(ea_meta, tm_meta, fb_meta)

    final_df = enrich_predictions(final_df, merged_df, master_meta)

    return final_df, merged_df, metrics_json, metrics_csv, all_runs_metrics, shap_df, master_meta


final_df, merged_df, metrics_json, metrics_csv, all_runs_metrics, shap_df, master_meta = load_data()

st.title("Football Player Market Value Prediction App")

with st.expander("Loaded files", expanded=False):
    st.write("App file:", str(APP_FILE))
    st.write("Project root:", str(PROJECT_ROOT))
    st.write("Data dir:", str(DATA_DIR))
    st.write("Final values:", str(FINAL_VALUES_FILE), FINAL_VALUES_FILE.exists())
    st.write("Merged inference dataset:", str(MERGED_FILE), MERGED_FILE.exists())
    st.write("Metrics JSON:", str(METRICS_JSON_FILE), METRICS_JSON_FILE.exists())
    st.write("Metrics CSV:", str(METRICS_CSV_FILE), METRICS_CSV_FILE.exists())
    st.write("All runs metrics:", str(ALL_RUNS_METRICS_FILE), ALL_RUNS_METRICS_FILE.exists())
    st.write("SHAP summary:", str(SHAP_FILE), SHAP_FILE.exists())
    st.write("EA raw:", str(EA_FILE), EA_FILE.exists())
    st.write("TM raw:", str(TM_FILE), TM_FILE.exists())
    st.write("FB raw:", str(FB_FILE), FB_FILE.exists())

if final_df is None or final_df.empty:
    st.error(
        "final_player_values.csv could not be found or loaded.\n\n"
        "Expected location:\n"
        f"{FINAL_VALUES_FILE}\n\n"
        "Run the modelling pipeline first, then launch this app with:\n"
        "streamlit run scripts/app.py"
    )
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filters")

position_options = ["All"] + sorted(final_df["position_group"].dropna().astype(str).unique().tolist())
selected_position = st.sidebar.selectbox("Position", position_options)

league_col = "club_league_name" if "club_league_name" in final_df.columns else None
if league_col is not None:
    league_options = ["All"] + sorted(final_df[league_col].dropna().astype(str).unique().tolist())
else:
    league_options = ["All"]

selected_league = st.sidebar.selectbox("League", league_options)

min_actual = 0
if "actual_value" in final_df.columns:
    min_actual = st.sidebar.number_input("Minimum actual value (€)", min_value=0, value=0, step=100000)

filtered_df = final_df.copy()

if selected_position != "All":
    filtered_df = filtered_df[filtered_df["position_group"].astype(str) == selected_position]

if selected_league != "All" and league_col is not None:
    filtered_df = filtered_df[filtered_df[league_col].astype(str) == selected_league]

if "actual_value" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["actual_value"].fillna(0) >= min_actual]

tabs = st.tabs([
    "Player Lookup",
    "Market Inefficiencies",
    "Uncertainty & Shortlists",
    "Insights Dashboard",
    "Metrics & Model"
])

# ============================================================
# TAB 1
# ============================================================

with tabs[0]:
    st.header("Player Lookup")

    # Use full enriched predictions file only
    player_options = sorted(filtered_df["player_name"].dropna().astype(str).unique().tolist())
    selected_player = st.selectbox("Select a player", player_options)

    row = filtered_df[filtered_df["player_name"].astype(str) == selected_player].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Value", format_eur(row.get("predicted_value_mean")))
    c2.metric("Actual Value", format_eur(row.get("actual_value")))
    c3.metric("Difference", format_eur(row.get("predicted_minus_actual")))
    c4.metric("% Difference", f"{row.get('percentage_diff', np.nan):.2f}%" if pd.notna(row.get("percentage_diff")) else "N/A")

    st.subheader("Profile")
    p1, p2, p3, p4 = st.columns(4)
    p1.write(f"**Position:** {row.get('position_group', 'N/A')}")
    p2.write(f"**League:** {row.get('club_league_name', 'N/A')}")
    p3.write(f"**Club:** {row.get('club_name', 'N/A')}")
    p4.write(f"**Status:** {row.get('valuation_label', 'N/A')}")

    p5, p6, p7, p8 = st.columns(4)
    p5.write(f"**Age:** {get_age_value(row)}")
    p6.write(f"**Overall Rating:** {row.get('overall_rating', 'N/A')}")
    p7.write(f"**Potential:** {row.get('potential', 'N/A')}")
    p8.write(f"**Wage:** {format_eur(row.get('wage'))}")

    p9, p10, p11, p12 = st.columns(4)
    p9.write(f"**Goal Contribution / 90:** {row.get('goal_contrib_per90', 'N/A')}")
    p10.write(f"**Minutes:** {row.get('Min', 'N/A')}")
    p11.write(f"**Contract Years Left:** {row.get('contract_years_left', row.get('contract_years_left_ea', 'N/A'))}")
    p12.write(f"**Player ID:** {row.get('player_id', 'N/A')}")

    if "predicted_value_std" in row.index and pd.notna(row.get("predicted_value_std")):
        lower = max(0, row["predicted_value_mean"] - row["predicted_value_std"])
        upper = row["predicted_value_mean"] + row["predicted_value_std"]
        st.write(f"**Prediction uncertainty (std):** {format_eur(row['predicted_value_std'])}")
        st.write(f"**Indicative range:** {format_eur(lower)} to {format_eur(upper)}")

    st.subheader("Explanation")
    for item in build_simple_explanation(row):
        st.write(f"- {item}")

# ============================================================
# TAB 2
# ============================================================

with tabs[1]:
    st.header("Market Inefficiencies")

    if "percentage_diff" not in filtered_df.columns:
        st.warning("percentage_diff is missing, so undervaluation / overvaluation tables cannot be shown.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 25 Undervalued")
            undervalued = filtered_df.sort_values("percentage_diff", ascending=False).head(25)
            st.dataframe(
                safe_column_subset(
                    undervalued,
                    ["player_name", "position_group", "club_name", "club_league_name",
                     "actual_value", "predicted_value_mean", "percentage_diff"]
                ),
                use_container_width=True
            )

        with col2:
            st.subheader("Top 25 Overvalued")
            overvalued = filtered_df.sort_values("percentage_diff", ascending=True).head(25)
            st.dataframe(
                safe_column_subset(
                    overvalued,
                    ["player_name", "position_group", "club_name", "club_league_name",
                     "actual_value", "predicted_value_mean", "percentage_diff"]
                ),
                use_container_width=True
            )

        st.subheader("League-Based Market Inefficiencies")
        if "club_league_name" in filtered_df.columns:
            league_summary = (
                filtered_df.groupby("club_league_name", as_index=False)
                .agg(
                    avg_pct_diff=("percentage_diff", "mean"),
                    avg_actual_value=("actual_value", "mean"),
                    avg_predicted_value=("predicted_value_mean", "mean"),
                    player_count=("player_name", "count")
                )
                .sort_values("avg_pct_diff", ascending=False)
            )
            st.dataframe(league_summary, use_container_width=True)

            fig_league = px.bar(
                league_summary,
                x="club_league_name",
                y="avg_pct_diff",
                title="Average Percentage Difference by League"
            )
            st.plotly_chart(fig_league, use_container_width=True)

# ============================================================
# TAB 3
# ============================================================

with tabs[2]:
    st.header("Uncertainty and Shortlists")

    if "predicted_value_std" in filtered_df.columns:
        st.subheader("Most uncertain players")
        uncertain = filtered_df.sort_values("predicted_value_std", ascending=False).head(25)
        st.dataframe(
            safe_column_subset(
                uncertain,
                ["player_name", "position_group", "club_name", "club_league_name",
                 "predicted_value_mean", "predicted_value_std", "percentage_diff"]
            ),
            use_container_width=True
        )

        fig_uncertainty = px.histogram(
            filtered_df,
            x="predicted_value_std",
            nbins=40,
            title="Distribution of Prediction Uncertainty"
        )
        st.plotly_chart(fig_uncertainty, use_container_width=True)
    else:
        st.info("predicted_value_std is not available.")

# ============================================================
# TAB 4
# ============================================================

with tabs[3]:
    st.header("Insights Dashboard")

    if "actual_value" in filtered_df.columns and "predicted_value_mean" in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x="actual_value",
            y="predicted_value_mean",
            color="position_group",
            hover_data=["player_name"],
            title="Actual vs Predicted Player Values"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    if "percentage_diff" in filtered_df.columns:
        error_by_position = (
            filtered_df.groupby("position_group", as_index=False)["percentage_diff"]
            .mean()
            .sort_values("percentage_diff", ascending=False)
        )
        fig_pos = px.bar(
            error_by_position,
            x="position_group",
            y="percentage_diff",
            title="Average Percentage Difference by Position"
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    if "club_league_name" in filtered_df.columns and "percentage_diff" in filtered_df.columns:
        league_view = (
            filtered_df.groupby("club_league_name", as_index=False)
            .agg(
                avg_pct_diff=("percentage_diff", "mean"),
                avg_predicted_value=("predicted_value_mean", "mean"),
                player_count=("player_name", "count")
            )
            .sort_values("avg_pct_diff", ascending=False)
        )
        st.subheader("League-level view")
        st.dataframe(league_view, use_container_width=True)

    if shap_df is not None and not shap_df.empty:
        st.subheader("Top SHAP Features")
        shap_display = shap_df.copy()
        if "mean_abs_shap_mean" in shap_display.columns and "mean_abs_shap" not in shap_display.columns:
            shap_display = shap_display.rename(columns={"mean_abs_shap_mean": "mean_abs_shap"})

        if {"feature", "mean_abs_shap"}.issubset(shap_display.columns):
            top20 = shap_display.head(20).sort_values("mean_abs_shap", ascending=True)
            fig_shap = px.bar(
                top20,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Top 20 Features by Mean SHAP Importance"
            )
            st.plotly_chart(fig_shap, use_container_width=True)

# ============================================================
# TAB 5
# ============================================================

with tabs[4]:
    st.header("Metrics and Model Information")

    st.subheader("Metric Summary")
    if metrics_json is not None:
        rows = []
        for metric, vals in metrics_json.items():
            rows.append({
                "metric": metric,
                "mean": vals.get("mean"),
                "std": vals.get("std"),
                "min": vals.get("min"),
                "max": vals.get("max")
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    elif metrics_csv is not None:
        st.dataframe(metrics_csv, use_container_width=True)
    else:
        st.info("No metrics summary file found.")

    st.subheader("All Runs Metrics")
    if all_runs_metrics is not None:
        st.dataframe(all_runs_metrics, use_container_width=True)
    else:
        st.info("No all_runs_metrics.csv file found.")

    st.subheader("Interpretation")
    st.write("""
    - Positive percentage difference means the model believes the player may be undervalued relative to the observed market value.
    - Negative percentage difference means the model believes the player may be overvalued.
    - Higher prediction standard deviation means the valuation is less stable across repeated runs.
    """)