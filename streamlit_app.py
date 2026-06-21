import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# PATH SETUP
# ============================================================
# All runtime data lives in the data/ folder next to this file, so the app
# behaves identically locally and on Streamlit Community Cloud.

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(page_title="Football Player Valuation App", layout="wide")

# Model outputs
FINAL_VALUES_FILE = DATA_DIR / "final_player_values.csv"
MERGED_FILE = DATA_DIR / "merged_inference_dataset.csv"
METRICS_JSON_FILE = DATA_DIR / "metrics_summary.json"
METRICS_CSV_FILE = DATA_DIR / "metrics_summary.csv"
ALL_RUNS_METRICS_FILE = DATA_DIR / "all_runs_metrics.csv"
SHAP_FILE = DATA_DIR / "shap_summary.csv"

# Cleaned source datasets used to enrich player metadata
EA_FILE = DATA_DIR / "player_stats_cleaned.csv"
TM_FILE = DATA_DIR / "transfermarkt_merged_players_with_valuation.csv"
FB_FILE = DATA_DIR / "Fbref_Final_Data.csv"

# Experiment history (written by scripts/experiment_tracker.py)
EXPERIMENT_LOG_FILE = APP_DIR / "experiments" / "experiment_log.csv"


# ============================================================
# APPEARANCE & ACCESSIBILITY
# ============================================================
# Okabe-Ito: a colour-blind-safe qualitative palette used for every chart so
# categories stay distinguishable for users with colour-vision deficiencies.
OKABE_ITO = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]

APPEARANCE_THEMES = {
    "Light": {
        "bg": "#ffffff", "secondary_bg": "#f3f5f7", "text": "#1a1a1a",
        "primary": "#1b9e77", "plotly_template": "plotly_white",
    },
    "Dark": {
        "bg": "#0e1117", "secondary_bg": "#1c2027", "text": "#e8e8e8",
        "primary": "#2dd4a7", "plotly_template": "plotly_dark",
    },
    "High contrast": {
        "bg": "#000000", "secondary_bg": "#0a0a0a", "text": "#ffffff",
        "primary": "#ffd400", "plotly_template": "plotly_dark",
    },
}

# Multipliers applied to the root font size for readability.
TEXT_SCALES = {"Normal": 1.0, "Large": 1.15, "Extra large": 1.3}


def apply_appearance(theme, scale):
    """Restyle the app chrome via CSS and set colour-blind-safe chart defaults."""
    base_px = 16 * scale
    st.markdown(
        f"""
        <style>
        html {{ font-size: {base_px:.1f}px; }}
        .stApp {{ background-color: {theme['bg']}; color: {theme['text']}; }}
        [data-testid="stHeader"] {{ background-color: {theme['bg']}; }}
        [data-testid="stSidebar"] {{ background-color: {theme['secondary_bg']}; }}
        .stApp p, .stApp li, .stApp label, .stApp span,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stTabs [data-baseweb="tab"] {{ color: {theme['text']}; }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {theme['text']}; }}
        /* Visible keyboard-focus outline for accessibility */
        .stApp a:focus, .stApp button:focus, .stApp input:focus,
        .stApp select:focus, .stApp [role="tab"]:focus {{
            outline: 3px solid {theme['primary']}; outline-offset: 2px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    px.defaults.template = theme["plotly_template"]
    px.defaults.color_discrete_sequence = OKABE_ITO


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


def safe_column_subset(df, cols):
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def get_today():
    return pd.Timestamp.today().normalize()


def calculate_live_age_from_dob(dob):
    if pd.isna(dob):
        return np.nan
    today = get_today()
    dob = pd.to_datetime(dob, errors="coerce")
    if pd.isna(dob):
        return np.nan
    return round((today - dob).days / 365.25, 2)


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
    elif pd.notna(row.get("contract_years_left_ea")) and row.get("contract_years_left_ea", 0) >= 2:
        explanations.append("Longer contract length strengthens club leverage in the market.")

    age_value = row.get("live_age", np.nan)
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

    # deduplicate
    seen = set()
    deduped = []
    for e in explanations:
        if e not in seen:
            deduped.append(e)
            seen.add(e)

    return deduped


def deal_score(predicted_value, proposed_price):
    """
    Returns:
    - score from 0 to 100
    - difference value
    - percentage over/under predicted
    """
    if pd.isna(predicted_value) or pd.isna(proposed_price) or predicted_value <= 0:
        return np.nan, np.nan, np.nan

    diff = predicted_value - proposed_price
    pct = (diff / predicted_value) * 100

    # Very favorable deals should score near 100.
    # Overpaying significantly should push score toward 0.
    score = 50 + (pct * 2.0)
    score = max(0, min(100, score))

    return round(score, 1), diff, pct


def deal_label(score):
    if pd.isna(score):
        return "Unavailable"
    if score >= 85:
        return "Outstanding deal"
    if score >= 70:
        return "Very good deal"
    if score >= 55:
        return "Good / fair deal"
    if score >= 40:
        return "Slight overpay"
    if score >= 20:
        return "Poor deal"
    return "Very poor deal"


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
        ea["live_age"] = ea["dob"].apply(calculate_live_age_from_dob)

    if "club_contract_valid_until" in ea.columns:
        ea["club_contract_valid_until"] = pd.to_datetime(ea["club_contract_valid_until"], errors="coerce")
        ref_date = get_today()
        ea["contract_years_left_ea"] = ((ea["club_contract_valid_until"] - ref_date).dt.days / 365.25).clip(lower=0)

    # club info only from EA
    if "club_name" not in ea.columns:
        ea["club_name"] = np.nan

    keep_cols = [
        "name_key", "player_name", "dob", "live_age", "overall_rating", "potential", "wage",
        "club_name", "club_league_name", "contract_years_left_ea", "positions"
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

    if "date_of_birth" in tm.columns:
        tm["date_of_birth"] = pd.to_datetime(tm["date_of_birth"], errors="coerce")
        tm["live_age_tm"] = tm["date_of_birth"].apply(calculate_live_age_from_dob)

    # Carry the observed market value so it can back-fill actual_value for
    # players the model only ran inference on (e.g. Haaland, Mbappe).
    if "market_value_in_eur_valuation" in tm.columns:
        tm["actual_value_tm"] = pd.to_numeric(
            tm["market_value_in_eur_valuation"], errors="coerce"
        )

    keep_cols = [
        "name_key", "player_name", "date_of_birth", "live_age_tm",
        "age", "age_at_valuation", "contract_years_left", "contract_expiration_date",
        "sub_position", "actual_value_tm"
    ]
    keep_cols = [c for c in keep_cols if c in tm.columns]

    tm_meta = tm[keep_cols].copy()
    return tm_meta


def prepare_fb_metadata(fb):
    if fb is None or fb.empty:
        return None

    fb = fb.copy()

    if "Player" not in fb.columns:
        return None

    fb["player_name"] = fb["Player"]
    fb["name_key"] = fb["player_name"].apply(normalize_name)

    fb_numeric_cols = [
        "Age", "MP", "Starts", "Subs", "Min", "90s",
        "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt",
        "Sh", "SoT", "Int", "TklW",
        "Gls/90", "Sh/90", "SoT/90", "Int/90", "TklW/90", "SoT%"
    ]
    for col in fb_numeric_cols:
        if col in fb.columns:
            fb[col] = pd.to_numeric(fb[col], errors="coerce")

    if {"name_key", "Min"}.issubset(fb.columns):
        fb = fb.sort_values(["name_key", "Min"], ascending=[True, False]).drop_duplicates("name_key", keep="first")
    else:
        fb = fb.drop_duplicates("name_key", keep="first")

    if {"Gls", "Ast", "90s"}.issubset(fb.columns):
        fb["goal_contrib_per90"] = np.where(fb["90s"] > 0, (fb["Gls"] + fb["Ast"]) / fb["90s"], np.nan)

    keep_cols = [
        "name_key", "player_name", "Comp", "Pos", "Age", "Min", "90s",
        "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt",
        "Sh", "SoT", "Int", "TklW",
        "Gls/90", "Sh/90", "SoT/90", "Int/90", "TklW/90", "SoT%",
        "goal_contrib_per90"
    ]
    keep_cols = [c for c in keep_cols if c in fb.columns]

    fb_meta = fb[keep_cols].copy()
    return fb_meta


def build_master_metadata(ea_meta, tm_meta, fb_meta):
    frames = [x for x in [ea_meta, tm_meta, fb_meta] if x is not None and not x.empty]

    if not frames:
        return None

    master = frames[0].copy()

    for i, df in enumerate(frames[1:], start=2):
        df = df.copy()
        overlap = [c for c in df.columns if c in master.columns and c != "name_key"]
        rename_map = {c: f"{c}_src{i}" for c in overlap}
        df = df.rename(columns=rename_map)
        master = master.merge(df, on="name_key", how="outer")

    def coalesce_column(target_base_names, final_name):
        candidates = []
        for base in target_base_names:
            candidates.extend([c for c in master.columns if c == base or c.startswith(f"{base}_src")])
        candidates = list(dict.fromkeys(candidates))

        if not candidates:
            return

        if final_name not in master.columns:
            master[final_name] = master[candidates[0]]

        for c in candidates:
            if c != final_name:
                master[final_name] = master[final_name].fillna(master[c])

    coalesce_column(["player_name"], "player_name")
    coalesce_column(["live_age", "live_age_tm"], "live_age")
    coalesce_column(["overall_rating"], "overall_rating")
    coalesce_column(["potential"], "potential")
    coalesce_column(["wage"], "wage")
    coalesce_column(["club_name"], "club_name")
    coalesce_column(["club_league_name", "Comp"], "club_league_name")
    coalesce_column(["contract_years_left", "contract_years_left_ea"], "contract_years_left")
    coalesce_column(["contract_years_left_ea"], "contract_years_left_ea")
    coalesce_column(["positions", "sub_position", "Pos"], "position_group")
    coalesce_column(["actual_value_tm"], "actual_value_tm")

    stat_fields = [
        "Min", "90s", "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt",
        "Sh", "SoT", "Int", "TklW",
        "Gls/90", "Sh/90", "SoT/90", "Int/90", "TklW/90", "SoT%",
        "goal_contrib_per90"
    ]
    for stat in stat_fields:
        coalesce_column([stat], stat)

    master = master.drop_duplicates("name_key")
    return master


# Maps granular position codes (and the original coarse labels) to four buckets
# so the Position filter and per-position charts stay readable after the source
# data brings in detailed positions like "CDM, CM" or "LB, LM".
POSITION_GROUP_MAP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF", "WB": "DEF", "DEF": "DEF",
    "CDM": "MID", "DM": "MID", "CM": "MID", "CAM": "MID", "AM": "MID",
    "LM": "MID", "RM": "MID", "MID": "MID",
    "ST": "FWD", "CF": "FWD", "LW": "FWD", "RW": "FWD", "FW": "FWD", "FWD": "FWD",
}


def coarse_position(pos):
    """Reduce a (possibly multi-) position string to one of GK/DEF/MID/FWD/UNK."""
    if pd.isna(pos):
        return "UNK"
    first = str(pos).split(",")[0].strip().upper()
    return POSITION_GROUP_MAP.get(first, "UNK")


def recover_identity_from_id(df):
    """Derive name_key (and a fallback date of birth) from the composite player_id.

    player_id has the form '<normalized_name>__<dob>' and is present on EVERY
    row, whereas player_name is missing for inference-only players (e.g. Erling
    Haaland, Kylian Mbappe). Driving name_key from the id lets those rows match
    the source datasets and recover their real, accented names and metadata,
    instead of being dropped from the dashboard.
    """
    if df is None or df.empty or "player_id" not in df.columns:
        return df
    df = df.copy()
    parts = df["player_id"].astype("string").str.split("__", n=1)
    key = parts.str[0].str.strip()
    df["name_key"] = key.where(key.str.len() > 0)
    df["dob_from_id"] = pd.to_datetime(parts.str[1].str.strip(), errors="coerce")
    return df


def finalize_identity(df):
    """Last-resort fallbacks so every prediction has a readable name and age.

    Runs after source enrichment: any player_name still missing is filled from a
    title-cased name_key, and live_age is back-filled from the id-derived dob.
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    if "name_key" in df.columns:
        readable = df["name_key"].astype("string").str.title()
        if "player_name" in df.columns:
            pn = df["player_name"].astype("string")
            df["player_name"] = pn.where(pn.notna() & (pn.str.strip() != ""), readable)
        else:
            df["player_name"] = readable

    if "dob_from_id" in df.columns:
        age_from_id = df["dob_from_id"].apply(calculate_live_age_from_dob)
        if "live_age" in df.columns:
            df["live_age"] = df["live_age"].fillna(age_from_id)
        else:
            df["live_age"] = age_from_id

    # Keep the detailed position for display, but bucket position_group so the
    # filter and per-position charts remain usable.
    if "position_group" in df.columns:
        df["position_detail"] = df["position_group"]
        df["position_group"] = df["position_group"].apply(coarse_position)

    return df


def enrich_predictions(final_df, merged_df, master_meta):
    if final_df is None or final_df.empty:
        return final_df

    final_df = ensure_required_columns(final_df.copy())

    if "name_key" not in final_df.columns:
        final_df["name_key"] = final_df["player_name"].apply(normalize_name)

    if merged_df is not None and not merged_df.empty:
        merged_df = merged_df.copy()
        if "player_name" not in merged_df.columns:
            for c in ["full_name", "name"]:
                if c in merged_df.columns:
                    merged_df["player_name"] = merged_df[c]
                    break
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

    if master_meta is not None and not master_meta.empty:
        meta_keep = [
            "name_key", "player_name", "live_age", "position_group",
            "club_league_name", "club_name", "overall_rating", "potential", "wage",
            "contract_years_left", "contract_years_left_ea", "actual_value_tm",
            "goal_contrib_per90", "Min", "90s", "Gls", "Ast", "G+A", "G-PK",
            "PK", "PKatt", "Sh", "SoT", "Int", "TklW",
            "Gls/90", "Sh/90", "SoT/90", "Int/90", "TklW/90", "SoT%"
        ]
        meta_keep = [c for c in meta_keep if c in master_meta.columns]
        master_small = master_meta[meta_keep].drop_duplicates("name_key")

        final_df = final_df.merge(
            master_small,
            on="name_key",
            how="left",
            suffixes=("", "_meta")
        )

    def fill_from_sources(base_col):
        for src_col in [f"{base_col}_merged", f"{base_col}_meta"]:
            if src_col in final_df.columns:
                if base_col not in final_df.columns:
                    final_df[base_col] = np.nan
                final_df[base_col] = final_df[base_col].fillna(final_df[src_col])

    for col in [
        "player_name", "position_group", "club_league_name", "club_name",
        "overall_rating", "potential", "wage",
        "age_ea", "age", "age_at_valuation", "contract_years_left",
        "contract_years_left_ea", "goal_contrib_per90", "Min", "90s",
        "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "Sh", "SoT", "Int", "TklW",
        "Gls/90", "Sh/90", "SoT/90", "Int/90", "TklW/90", "SoT%"
    ]:
        fill_from_sources(col)

    if "live_age" not in final_df.columns:
        final_df["live_age"] = np.nan
    fill_from_sources("live_age")

    if "live_age" in final_df.columns:
        if "age_ea" in final_df.columns:
            final_df["live_age"] = final_df["live_age"].fillna(final_df["age_ea"])
        if "age" in final_df.columns:
            final_df["live_age"] = final_df["live_age"].fillna(final_df["age"])

    # Back-fill the observed market value for inference-only players from the TM
    # source, then drop the derived columns so they are recomputed from the now
    # more-complete actual_value. (Model-evaluated players keep their original
    # value; a full model re-run will later regenerate all of them consistently.)
    fill_from_sources("actual_value_tm")
    if "actual_value_tm" in final_df.columns:
        if "actual_value" not in final_df.columns:
            final_df["actual_value"] = np.nan
        final_df["actual_value"] = final_df["actual_value"].fillna(final_df["actual_value_tm"])
        final_df = final_df.drop(
            columns=[c for c in ["predicted_minus_actual", "percentage_diff", "valuation_label"]
                     if c in final_df.columns]
        )

    final_df = ensure_required_columns(final_df)

    return final_df


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    final_df = safe_read_csv(FINAL_VALUES_FILE)
    final_df = recover_identity_from_id(final_df)
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
    final_df = finalize_identity(final_df)

    return final_df, merged_df, metrics_json, metrics_csv, all_runs_metrics, shap_df


final_df, merged_df, metrics_json, metrics_csv, all_runs_metrics, shap_df = load_data()

st.title("Football Player Market Value Prediction App")
st.caption(
    "Predicted transfer-market values from a hybrid model, with deal evaluation, "
    "market-inefficiency analysis and prediction uncertainty."
)

if final_df is None or final_df.empty:
    st.error(
        "Player valuation data could not be loaded.\n\n"
        f"Expected file: `{FINAL_VALUES_FILE.name}` in the `data/` folder.\n\n"
        "If you are running locally, generate the model outputs first, then launch with:\n"
        "`streamlit run streamlit_app.py`"
    )
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Appearance")
theme_name = st.sidebar.selectbox(
    "Colour theme", list(APPEARANCE_THEMES.keys()),
    help="Restyles the interface. Charts always use a colour-blind-safe palette.",
)
text_size = st.sidebar.selectbox("Text size", list(TEXT_SCALES.keys()))
apply_appearance(APPEARANCE_THEMES[theme_name], TEXT_SCALES[text_size])
st.sidebar.caption(
    "For full dark mode including data tables, you can also use "
    "☰ → Settings → Theme."
)

st.sidebar.divider()
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

st.sidebar.caption(
    f"Matching players: {filtered_df['player_name'].nunique():,} "
    f"of {final_df['player_name'].nunique():,}"
)

tabs = st.tabs([
    "Player Lookup",
    "Deal Evaluator",
    "Market Inefficiencies",
    "Uncertainty",
    "Insights Dashboard",
    "Model Metrics"
])

# ============================================================
# TAB 1: PLAYER LOOKUP
# ============================================================

with tabs[0]:
    st.header("Player Lookup")

    player_options = sorted(filtered_df["player_name"].dropna().astype(str).unique().tolist())
    if not player_options:
        st.info("No players match the current filters. Adjust the filters in the sidebar.")
    else:
        selected_player = st.selectbox("Select a player", player_options)

        row = filtered_df[filtered_df["player_name"].astype(str) == selected_player].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Value", format_eur(row.get("predicted_value_mean")))
        c2.metric("Actual Value", format_eur(row.get("actual_value")))
        c3.metric("Difference", format_eur(row.get("predicted_minus_actual")))
        c4.metric("% Difference", f"{row.get('percentage_diff', np.nan):.2f}%" if pd.notna(row.get("percentage_diff")) else "N/A")

        st.subheader("Profile")
        p1, p2, p3, p4 = st.columns(4)
        p1.write(f"**Position:** {row.get('position_detail') or row.get('position_group', 'N/A')}")
        p2.write(f"**League:** {row.get('club_league_name', 'N/A')}")
        p3.write(f"**Club:** {row.get('club_name', 'N/A')}")
        p4.write(f"**Status:** {row.get('valuation_label', 'N/A')}")

        p5, p6, p7, p8 = st.columns(4)
        p5.write(f"**Current Age:** {row.get('live_age', 'N/A')}")
        p6.write(f"**Overall Rating:** {row.get('overall_rating', 'N/A')}")
        p7.write(f"**Potential:** {row.get('potential', 'N/A')}")
        p8.write(f"**Wage:** {format_eur(row.get('wage'))}")

        p9, p10, p11, p12 = st.columns(4)
        p9.write(f"**Minutes:** {row.get('Min', 'N/A')}")
        p10.write(f"**90s:** {row.get('90s', 'N/A')}")
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
# TAB 2: DEAL EVALUATOR
# ============================================================

with tabs[1]:
    st.header("Deal Evaluator")

    st.write("""
    Select a player and enter a proposed transfer price. The app compares that fee
    against the model’s predicted value and returns a **deal score from 0 to 100**.
    """)

    player_options = sorted(filtered_df["player_name"].dropna().astype(str).unique().tolist())
    if not player_options:
        st.info("No players match the current filters. Adjust the filters in the sidebar.")
    else:
        deal_player = st.selectbox("Choose player for deal evaluation", player_options, key="deal_player")

        deal_row = filtered_df[filtered_df["player_name"].astype(str) == deal_player].iloc[0]
        predicted_val = deal_row.get("predicted_value_mean", np.nan)

        proposed_price = st.number_input(
            "Proposed purchase price (€)",
            min_value=0,
            value=int(predicted_val) if pd.notna(predicted_val) else 1000000,
            step=500000
        )

        score, diff, pct = deal_score(predicted_val, proposed_price)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Predicted Value", format_eur(predicted_val))
        d2.metric("Proposed Price", format_eur(proposed_price))
        d3.metric("Difference vs Model", format_eur(diff))
        d4.metric("Deal Score", f"{score}/100" if pd.notna(score) else "N/A")

        st.subheader("Assessment")
        st.write(f"**Rating:** {deal_label(score)}")

        if pd.notna(pct):
            if pct > 0:
                st.write(f"The proposed fee is **{pct:.2f}% below** the modelled value, which suggests a relatively favourable deal.")
            elif pct < 0:
                st.write(f"The proposed fee is **{abs(pct):.2f}% above** the modelled value, which suggests an overpay relative to the model.")
            else:
                st.write("The proposed fee is exactly aligned with the modelled value.")

# ============================================================
# TAB 3: MARKET INEFFICIENCIES
# ============================================================

with tabs[2]:
    st.header("Market Inefficiencies")

    st.write("""
    **How to read this section**
    
    - A **positive percentage difference** means the model predicts a player should be worth **more** than their observed market value.  
      This suggests the player may be **undervalued**.
    - A **negative percentage difference** means the model predicts a player should be worth **less** than their observed market value.  
      This suggests the player may be **overvalued**.
    - League tables summarise the **average percentage difference** across players in each league.
    """)

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
                width="stretch"
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
                width="stretch"
            )

        st.subheader("League-Based Market Inefficiencies")

        # IMPORTANT: all leagues from the full dataset, not filtered_df
        if "club_league_name" in final_df.columns and "percentage_diff" in final_df.columns:
            league_summary = (
                final_df.groupby("club_league_name", dropna=False, as_index=False)
                .agg(
                    avg_pct_diff=("percentage_diff", "mean"),
                    avg_actual_value=("actual_value", "mean"),
                    avg_predicted_value=("predicted_value_mean", "mean"),
                    player_count=("player_name", "count")
                )
                .sort_values("avg_pct_diff", ascending=False)
            )

            st.dataframe(league_summary, width="stretch")

            if not league_summary.empty:
                top_under = league_summary.iloc[0]
                top_over = league_summary.iloc[-1]

                st.write(
                    f"**Key insight:** The league with the highest average positive difference is "
                    f"**{top_under['club_league_name']}** ({top_under['avg_pct_diff']:.2f}%), "
                    f"suggesting players there may be relatively undervalued in aggregate."
                )
                st.write(
                    f"**Key insight:** The league with the most negative average difference is "
                    f"**{top_over['club_league_name']}** ({top_over['avg_pct_diff']:.2f}%), "
                    f"suggesting players there may be relatively overvalued in aggregate."
                )

            fig_league = px.bar(
                league_summary,
                y="club_league_name",
                x="avg_pct_diff",
                orientation="h",
                title="Average Percentage Difference by League"
            )
            st.plotly_chart(fig_league, width="stretch")

# ============================================================
# TAB 4: UNCERTAINTY
# ============================================================

with tabs[3]:
    st.header("Uncertainty")

    if "predicted_value_std" in filtered_df.columns:
        st.write("""
        This view shows how stable player valuations are across repeated model runs.
        Higher uncertainty means the model is less consistent about the player’s value.
        """)

        fig_uncertainty_scatter = px.scatter(
            filtered_df,
            x="predicted_value_mean",
            y="predicted_value_std",
            color="position_group",
            hover_data=["player_name", "club_name", "club_league_name"],
            title="Predicted Value vs Prediction Uncertainty",
            height=700
        )
        st.plotly_chart(fig_uncertainty_scatter, width="stretch")

        fig_uncertainty_hist = px.histogram(
            filtered_df,
            x="predicted_value_std",
            nbins=40,
            title="Distribution of Prediction Uncertainty"
        )
        st.plotly_chart(fig_uncertainty_hist, width="stretch")
    else:
        st.info("predicted_value_std is not available.")

# ============================================================
# TAB 5: INSIGHTS DASHBOARD
# ============================================================

with tabs[4]:
    st.header("Insights Dashboard")

    st.write("""
    **How to interpret the dashboard**
    
    - The scatter plot compares actual and predicted values. A tighter diagonal pattern indicates better calibration.
    - Position and league summaries highlight where the model tends to over- or under-estimate values.
    - If SHAP data is available, it shows which features matter most globally.
    """)

    if "actual_value" in filtered_df.columns and "predicted_value_mean" in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x="actual_value",
            y="predicted_value_mean",
            color="position_group",
            hover_data=["player_name", "club_name", "club_league_name"],
            title="Actual vs Predicted Player Values"
        )
        st.plotly_chart(fig_scatter, width="stretch")

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
        st.plotly_chart(fig_pos, width="stretch")

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
        st.subheader("League-Level View")
        st.dataframe(league_view, width="stretch")

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
            st.plotly_chart(fig_shap, width="stretch")

# ============================================================
# TAB 6: MODEL METRICS
# ============================================================

with tabs[5]:
    st.header("Model Metrics")

    st.write("""
    **How to interpret these metrics**
    
    - **R²** indicates how much variation in market value the model explains.
    - **RMSE** and **MAE** measure error in euro terms.
    - **Accuracy@10%** and **Accuracy@20%** show how often predictions fall close to the observed value.
    - Repeated-run averaging improves robustness and allows uncertainty estimation.
    """)

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
        st.dataframe(pd.DataFrame(rows), width="stretch")
    elif metrics_csv is not None:
        st.dataframe(metrics_csv, width="stretch")
    else:
        st.info("No metrics summary file found.")

    st.subheader("All Runs Metrics")
    if all_runs_metrics is not None:
        st.dataframe(all_runs_metrics, width="stretch")
    else:
        st.info("No all_runs_metrics.csv file found.")

    st.subheader("Experiment History")
    st.write(
        "Headline metrics recorded each time the model is run, so performance "
        "can be compared over time."
    )
    exp_history = safe_read_csv(EXPERIMENT_LOG_FILE)
    if exp_history is None or exp_history.empty:
        st.info(
            "No experiments logged yet. After a model run, record one with "
            "`python scripts/experiment_tracker.py log --readme`."
        )
    else:
        st.caption(f"{len(exp_history):,} logged experiment(s) — all-time averages:")
        a1, a2, a3, a4 = st.columns(4)
        if "R2" in exp_history.columns:
            a1.metric("Avg R²", f"{pd.to_numeric(exp_history['R2'], errors='coerce').mean():.3f}")
        if "MAE" in exp_history.columns:
            a2.metric("Avg MAE", format_eur(pd.to_numeric(exp_history["MAE"], errors="coerce").mean()))
        if "RMSE" in exp_history.columns:
            a3.metric("Avg RMSE", format_eur(pd.to_numeric(exp_history["RMSE"], errors="coerce").mean()))
        if "Accuracy@20%" in exp_history.columns:
            a4.metric("Avg Acc@20%", f"{pd.to_numeric(exp_history['Accuracy@20%'], errors='coerce').mean():.1f}%")

        for metric in ["R2", "MAE"]:
            if "experiment_id" in exp_history.columns and metric in exp_history.columns:
                fig_hist = px.line(
                    exp_history, x="experiment_id", y=metric, markers=True,
                    title=f"{metric} across experiments"
                )
                st.plotly_chart(fig_hist, width="stretch")

        st.dataframe(exp_history, width="stretch")