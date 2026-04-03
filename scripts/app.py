import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Football Player Valuation App",
    layout="wide"
)

DATA_DIR = "."
FINAL_VALUES_CANDIDATES = [
    "final_player_values.csv",
    "all_runs_predictions_wide.csv",
    "hybrid_predicted_player_values.csv",
    "predicted_player_values.csv"
]
MERGED_DATA_CANDIDATES = [
    "merged_inference_dataset.csv",
    "merged_dataset.csv",
    "player_stats_cleaned.csv"  # fallback only
]
SHAP_CANDIDATES = [
    "hybrid_shap_feature_importance.csv",
    "shap_summary.csv",
    "shap_fully_merged.csv",
    "shap_fallback.csv"
]
METRICS_CANDIDATES = [
    "metrics_summary.json",
    "hybrid_metrics.json",
    "metrics.json"
]
ARCHITECTURE_IMAGE_CANDIDATES = [
    "model_architecture_publication.png",
    "model_architecture.png"
]


# ============================================================
# FILE HELPERS
# ============================================================

def find_first_existing(candidates):
    for file_name in candidates:
        path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(path):
            return path
    return None


def safe_read_csv(path):
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_read_json(path):
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_all_data():
    final_values_path = find_first_existing(FINAL_VALUES_CANDIDATES)
    merged_data_path = find_first_existing(MERGED_DATA_CANDIDATES)
    shap_path = find_first_existing(SHAP_CANDIDATES)
    metrics_path = find_first_existing(METRICS_CANDIDATES)
    architecture_image_path = find_first_existing(ARCHITECTURE_IMAGE_CANDIDATES)

    final_df = safe_read_csv(final_values_path)
    merged_df = safe_read_csv(merged_data_path)
    shap_df = safe_read_csv(shap_path)
    metrics_json = safe_read_json(metrics_path)

    return {
        "final_df": final_df,
        "merged_df": merged_df,
        "shap_df": shap_df,
        "metrics_json": metrics_json,
        "final_values_path": final_values_path,
        "merged_data_path": merged_data_path,
        "shap_path": shap_path,
        "metrics_path": metrics_path,
        "architecture_image_path": architecture_image_path,
    }


# ============================================================
# PREPARATION
# ============================================================

def normalize_player_name_col(df):
    """
    Create/standardise a player_name column.
    """
    if df is None:
        return df

    df = df.copy()

    candidate_cols = [
        "player_name", "full_name", "name", "Player", "player", "player_x"
    ]
    for col in candidate_cols:
        if col in df.columns:
            df["player_name"] = df[col]
            return df

    df["player_name"] = "Unknown Player"
    return df


def normalize_position_col(df):
    if df is None:
        return df

    df = df.copy()

    candidate_cols = [
        "position_group", "positions", "Pos", "position", "sub_position"
    ]
    for col in candidate_cols:
        if col in df.columns:
            df["display_position"] = df[col]
            return df

    df["display_position"] = "Unknown"
    return df


def ensure_player_id(df):
    if df is None:
        return df

    df = df.copy()

    if "player_id" not in df.columns:
        if "player_name" not in df.columns:
            df = normalize_player_name_col(df)
        df["player_id"] = df["player_name"].astype(str)

    return df


def add_percentage_diff(df):
    if df is None:
        return df

    df = df.copy()

    if "actual_value" in df.columns:
        predicted_col = None
        for col in ["predicted_value_mean", "predicted_value", "predicted_value_median"]:
            if col in df.columns:
                predicted_col = col
                break

        if predicted_col is not None:
            df["percentage_diff"] = (
                (df[predicted_col] - df["actual_value"]) / df["actual_value"].replace(0, np.nan)
            ) * 100

            df["valuation_label"] = np.where(
                df["percentage_diff"] > 0,
                "Undervalued",
                "Overvalued"
            )

    return df


def merge_metadata(final_df, merged_df):
    """
    Merge output prediction table with metadata from modelling dataset.
    """
    if final_df is None:
        return None

    final_df = normalize_player_name_col(final_df)
    final_df = normalize_position_col(final_df)
    final_df = ensure_player_id(final_df)
    final_df = add_percentage_diff(final_df)

    if merged_df is None:
        return final_df

    merged_df = normalize_player_name_col(merged_df)
    merged_df = ensure_player_id(merged_df)

    useful_cols = [
        "player_id",
        "player_name",
        "position_group",
        "positions",
        "club_league_name",
        "country_name",
        "country_of_citizenship",
        "overall_rating",
        "potential",
        "wage",
        "age_ea",
        "age",
        "age_at_valuation",
        "contract_years_left",
        "contract_years_left_ea",
        "goal_contrib_per90",
        "Gls/90",
        "Sh/90",
        "SoT/90",
        "Int/90",
        "90s",
        "Min",
        "value",
        "market_value_in_eur_valuation",
    ]
    useful_cols = [c for c in useful_cols if c in merged_df.columns]

    if not useful_cols:
        return final_df

    metadata = merged_df[useful_cols].copy()

    if "player_id" not in metadata.columns:
        metadata["player_id"] = metadata["player_name"]

    metadata = metadata.drop_duplicates("player_id")

    out = final_df.merge(
        metadata,
        on="player_id",
        how="left",
        suffixes=("", "_meta")
    )

    if "player_name_meta" in out.columns:
        out["player_name"] = out["player_name"].fillna(out["player_name_meta"])

    if "position_group" not in out.columns and "position_group_meta" in out.columns:
        out["position_group"] = out["position_group_meta"]

    out = normalize_position_col(out)
    out = add_percentage_diff(out)

    return out


def get_prediction_column(df):
    if df is None:
        return None
    for col in ["predicted_value_mean", "predicted_value", "predicted_value_median"]:
        if col in df.columns:
            return col
    return None


def build_simple_explanation(row):
    explanations = []

    if pd.notna(row.get("overall_rating")) and row.get("overall_rating", 0) >= 80:
        explanations.append("High overall rating is a strong positive driver of predicted value.")

    if pd.notna(row.get("potential")) and row.get("potential", 0) >= 85:
        explanations.append("High potential increases the player’s long-term valuation.")

    if pd.notna(row.get("goal_contrib_per90")) and row.get("goal_contrib_per90", 0) >= 0.50:
        explanations.append("Strong goal contribution per 90 suggests efficient attacking output.")

    if pd.notna(row.get("contract_years_left")) and row.get("contract_years_left", 0) >= 2:
        explanations.append("Longer remaining contract length tends to increase market leverage and value.")
    elif pd.notna(row.get("contract_years_left_ea")) and row.get("contract_years_left_ea", 0) >= 2:
        explanations.append("Longer remaining contract length tends to increase market leverage and value.")

    age_value = np.nan
    for age_col in ["age_at_valuation", "age_ea", "age"]:
        if pd.notna(row.get(age_col)):
            age_value = row.get(age_col)
            break
    if pd.notna(age_value) and 23 <= age_value <= 28:
        explanations.append("The player is close to the typical peak-value age range.")
    elif pd.notna(age_value) and age_value < 21:
        explanations.append("Youth may increase upside, though current value can depend on uncertainty and playing time.")
    elif pd.notna(age_value) and age_value > 30:
        explanations.append("Older age may reduce long-term resale value despite current quality.")

    if pd.notna(row.get("Min")) and row.get("Min", 0) >= 1800:
        explanations.append("High minutes played suggest sustained involvement and stronger information quality.")

    if pd.notna(row.get("wage")) and row.get("wage", 0) > 0:
        explanations.append("Wage level may proxy perceived status and market standing.")

    if not explanations:
        explanations.append("This valuation appears to be driven by a mixed combination of ability, performance, age, and context.")

    return explanations


def format_eur(x):
    if pd.isna(x):
        return "N/A"
    return f"€{x:,.0f}"


# ============================================================
# APP UI
# ============================================================

data = load_all_data()

final_df = data["final_df"]
merged_df = data["merged_df"]
shap_df = data["shap_df"]
metrics_json = data["metrics_json"]
architecture_image_path = data["architecture_image_path"]

display_df = merge_metadata(final_df, merged_df)

st.title("Football Player Market Value Prediction Interface")

with st.expander("Loaded files", expanded=False):
    st.write("Predictions file:", data["final_values_path"])
    st.write("Merged metadata file:", data["merged_data_path"])
    st.write("SHAP file:", data["shap_path"])
    st.write("Metrics file:", data["metrics_path"])

if display_df is None or display_df.empty:
    st.error("No usable prediction file was found. Place final_player_values.csv or all_runs_predictions_wide.csv in the same folder as this app.")
    st.stop()

prediction_col = get_prediction_column(display_df)
if prediction_col is None:
    st.error("No prediction column found. Expected one of: predicted_value_mean, predicted_value, predicted_value_median.")
    st.stop()

display_df = normalize_player_name_col(display_df)
display_df = normalize_position_col(display_df)
display_df = ensure_player_id(display_df)
display_df = add_percentage_diff(display_df)

# Sidebar filters
st.sidebar.header("Filters")

position_options = ["All"] + sorted(display_df["display_position"].dropna().astype(str).unique().tolist())
selected_position = st.sidebar.selectbox("Position", position_options)

if "club_league_name" in display_df.columns:
    league_options = ["All"] + sorted(display_df["club_league_name"].dropna().astype(str).unique().tolist())
else:
    league_options = ["All"]
selected_league = st.sidebar.selectbox("League", league_options)

if "actual_value" in display_df.columns:
    min_actual_value = st.sidebar.number_input("Minimum actual value (€)", min_value=0, value=0, step=100000)
else:
    min_actual_value = 0

filtered_df = display_df.copy()

if selected_position != "All":
    filtered_df = filtered_df[filtered_df["display_position"].astype(str) == selected_position]

if selected_league != "All" and "club_league_name" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["club_league_name"].astype(str) == selected_league]

if "actual_value" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["actual_value"].fillna(0) >= min_actual_value]

tabs = st.tabs([
    "Player Lookup",
    "Market Inefficiencies",
    "Insights Dashboard",
    "Explainability",
    "Model Summary"
])

# ============================================================
# TAB 1: PLAYER LOOKUP
# ============================================================

with tabs[0]:
    st.header("Player Lookup")

    player_options = sorted(filtered_df["player_name"].dropna().astype(str).unique().tolist())
    selected_player = st.selectbox("Select a player", player_options)

    player_rows = filtered_df[filtered_df["player_name"].astype(str) == selected_player].copy()

    if player_rows.empty:
        st.warning("No player found.")
    else:
        row = player_rows.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Predicted Value", format_eur(row.get(prediction_col)))
        col2.metric("Actual Value", format_eur(row.get("actual_value")))
        if pd.notna(row.get(prediction_col)) and pd.notna(row.get("actual_value")):
            col3.metric("Difference", format_eur(row.get(prediction_col) - row.get("actual_value")))
        else:
            col3.metric("Difference", "N/A")
        col4.metric("% Difference", f"{row.get('percentage_diff', np.nan):.2f}%" if pd.notna(row.get("percentage_diff")) else "N/A")

        st.subheader("Player Profile")
        info_cols = st.columns(4)
        info_cols[0].write(f"**Position:** {row.get('display_position', 'N/A')}")
        info_cols[1].write(f"**League:** {row.get('club_league_name', 'N/A')}")
        info_cols[2].write(f"**Age:** {row.get('age_at_valuation', row.get('age_ea', row.get('age', 'N/A')))}")
        info_cols[3].write(f"**Valuation Status:** {row.get('valuation_label', 'N/A')}")

        info_cols2 = st.columns(4)
        info_cols2[0].write(f"**Overall Rating:** {row.get('overall_rating', 'N/A')}")
        info_cols2[1].write(f"**Potential:** {row.get('potential', 'N/A')}")
        info_cols2[2].write(f"**Goal Contribution / 90:** {row.get('goal_contrib_per90', 'N/A')}")
        info_cols2[3].write(f"**Minutes:** {row.get('Min', 'N/A')}")

        if "predicted_value_std" in row.index:
            st.write(f"**Prediction uncertainty (std across runs):** {format_eur(row.get('predicted_value_std'))}")

        st.subheader("Model Explanation")
        explanations = build_simple_explanation(row)
        for e in explanations:
            st.write(f"- {e}")

        if "predicted_value_std" in row.index and pd.notna(row.get("predicted_value_std")):
            lower = max(0, row.get(prediction_col) - row.get("predicted_value_std"))
            upper = row.get(prediction_col) + row.get("predicted_value_std")
            st.write(f"**Indicative prediction band:** {format_eur(lower)} to {format_eur(upper)}")

# ============================================================
# TAB 2: MARKET INEFFICIENCIES
# ============================================================

with tabs[1]:
    st.header("Most Overvalued and Undervalued Players")

    if "percentage_diff" not in filtered_df.columns:
        st.warning("Percentage difference could not be computed because actual values are missing.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most Undervalued")
            undervalued = filtered_df.sort_values("percentage_diff", ascending=False).head(25)
            st.dataframe(
                undervalued[[
                    "player_name", "display_position",
                    "actual_value", prediction_col, "percentage_diff"
                ]].rename(columns={
                    "display_position": "position",
                    prediction_col: "predicted_value"
                }),
                use_container_width=True
            )

        with col2:
            st.subheader("Most Overvalued")
            overvalued = filtered_df.sort_values("percentage_diff", ascending=True).head(25)
            st.dataframe(
                overvalued[[
                    "player_name", "display_position",
                    "actual_value", prediction_col, "percentage_diff"
                ]].rename(columns={
                    "display_position": "position",
                    prediction_col: "predicted_value"
                }),
                use_container_width=True
            )

        st.subheader("Downloadable filtered table")
        st.dataframe(filtered_df, use_container_width=True)

# ============================================================
# TAB 3: INSIGHTS DASHBOARD
# ============================================================

with tabs[2]:
    st.header("Insights Dashboard")

    if "actual_value" in filtered_df.columns and prediction_col in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x="actual_value",
            y=prediction_col,
            color="display_position",
            hover_data=["player_name"],
            title="Actual vs Predicted Player Values"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    if "predicted_value_std" in filtered_df.columns:
        fig_uncertainty = px.histogram(
            filtered_df,
            x="predicted_value_std",
            nbins=40,
            title="Distribution of Prediction Uncertainty"
        )
        st.plotly_chart(fig_uncertainty, use_container_width=True)

    if "percentage_diff" in filtered_df.columns:
        group_col = "display_position"
        error_by_position = filtered_df.groupby(group_col, as_index=False)["percentage_diff"].mean()
        fig_pos = px.bar(
            error_by_position,
            x=group_col,
            y="percentage_diff",
            title="Average Percentage Difference by Position"
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    if "club_league_name" in filtered_df.columns and "percentage_diff" in filtered_df.columns:
        league_view = (
            filtered_df.groupby("club_league_name", as_index=False)
            .agg(
                avg_pct_diff=("percentage_diff", "mean"),
                avg_predicted_value=(prediction_col, "mean"),
                player_count=("player_id", "count")
            )
            .sort_values("avg_pct_diff", ascending=False)
        )
        st.subheader("League-level valuation view")
        st.dataframe(league_view, use_container_width=True)

# ============================================================
# TAB 4: EXPLAINABILITY
# ============================================================

with tabs[3]:
    st.header("Explainability")

    if shap_df is None or shap_df.empty:
        st.info("No SHAP summary file was found. Place hybrid_shap_feature_importance.csv or shap_summary.csv in the app folder.")
    else:
        shap_df = shap_df.copy()

        if "mean_abs_shap_mean" in shap_df.columns and "mean_abs_shap" not in shap_df.columns:
            shap_df["mean_abs_shap"] = shap_df["mean_abs_shap_mean"]

        required = {"feature", "mean_abs_shap"}
        if not required.issubset(set(shap_df.columns)):
            st.warning("SHAP file does not contain the expected columns: feature, mean_abs_shap.")
        else:
            shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)

            st.subheader("Top 20 Global SHAP Features")
            top20 = shap_df.head(20).sort_values("mean_abs_shap", ascending=True)

            fig_shap = px.bar(
                top20,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Top 20 Features by Mean Absolute SHAP Value"
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            st.subheader("Top 10 SHAP Table")
            st.dataframe(shap_df.head(10), use_container_width=True)

            cumulative = shap_df.copy()
            cumulative["share"] = cumulative["mean_abs_shap"] / cumulative["mean_abs_shap"].sum()
            cumulative["cumulative_share"] = cumulative["share"].cumsum()
            cumulative["rank"] = range(1, len(cumulative) + 1)

            fig_cum = px.line(
                cumulative,
                x="rank",
                y="cumulative_share",
                title="Cumulative SHAP Importance"
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            st.subheader("Interpretation")
            st.write(
                "These SHAP values show which features most strongly influence the boosted meta-model on average. "
                "High mean absolute SHAP values indicate features that consistently move predicted market values."
            )

# ============================================================
# TAB 5: MODEL SUMMARY
# ============================================================

with tabs[4]:
    st.header("Model Summary")

    if architecture_image_path is not None and os.path.exists(architecture_image_path):
        st.image(architecture_image_path, caption="Hybrid multi-layer ensemble architecture")

    st.subheader("Architecture")
    st.write("""
    This application is based on a hybrid multi-layer ensemble architecture:

    **Layer 1: Base experts**
    - LightGBM on EA Sports FC features
    - XGBoost on FBref features
    - CatBoost on Transfermarkt/context features

    **Layer 2: Blending**
    - Mean prediction
    - Median prediction
    - Weighted prediction
    - Disagreement features between expert outputs

    **Layer 3: Meta learning**
    - Boosted meta model
    - Neural meta model (MLP)

    **Layer 4: Final output**
    - Weighted blend of the boosted and neural meta models

    Final player values are averaged across repeated runs when available.
    """)

    if metrics_json is not None:
        st.subheader("Evaluation Metrics")
        if isinstance(metrics_json, dict):
            metric_rows = []
            for key, value in metrics_json.items():
                if isinstance(value, dict):
                    metric_rows.append({
                        "metric": key,
                        "mean": value.get("mean"),
                        "std": value.get("std"),
                        "min": value.get("min"),
                        "max": value.get("max")
                    })
                else:
                    metric_rows.append({
                        "metric": key,
                        "value": value
                    })
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

    st.subheader("How to interpret market inefficiency")
    st.write("""
    - **Positive percentage difference**: the model predicts a higher value than the observed market value, suggesting the player may be **undervalued**.
    - **Negative percentage difference**: the model predicts a lower value than the observed market value, suggesting the player may be **overvalued**.
    """)

    st.subheader("Suggested extensions")
    st.write("""
    Future extensions could include:
    - local SHAP explanations per player
    - club-specific recruitment shortlists
    - league-specific dashboards
    - position-specific models
    - deployment to Streamlit Cloud
    """)