"""Daily model pipeline entry point (used by .github/workflows/daily-model.yml).

Leakage-safe hybrid football market-value model with a 15-run repeated ensemble.

Architecture
------------
  Layer 1  Base experts:   LightGBM (EA) + XGBoost (FBref) + CatBoost (TM)
  Layer 2  Blended experts: mean / median / weighted + disagreement features
  Layer 3  Meta models:    boosted meta (LightGBM) + neural meta (MLPRegressor)
  Layer 4  Final blend:    inverse-RMSE weighted blend of the two meta models

Leakage columns excluded from predictors: value, market_value_in_eur_valuation,
release_clause. Target = average(EA value, Transfermarkt value) when both exist.

What this script does, in order:
  1. train_model()  -> retrains and writes the output files into data/:
        final_player_values.csv, metrics_summary.json, metrics_summary.csv,
        all_runs_metrics.csv, shap_summary.csv
  2. log_experiment() -> records the run and archives its outputs.
  3. promote_best()   -> copies the best run (highest R2) into data/ for display.
  4. prune() + update_readme().

Run locally:

    python scripts/run_pipeline.py --notes "manual run"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Optional modelling packages
# ------------------------------------------------------------
USE_LIGHTGBM = True
USE_XGBOOST = True
USE_CATBOOST = True
USE_SHAP = True

try:
    from lightgbm import LGBMRegressor
except Exception:
    USE_LIGHTGBM = False

try:
    from xgboost import XGBRegressor
except Exception:
    USE_XGBOOST = False

try:
    from catboost import CatBoostRegressor
except Exception:
    USE_CATBOOST = False

try:
    import shap
except Exception:
    USE_SHAP = False

# Workaround for environments where packages expect scipy._lib to be loaded.
try:
    import scipy
    import scipy._lib  # noqa: F401
except Exception:
    pass

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Experiment tracking lives next to this file.
sys.path.append(str(Path(__file__).resolve().parent))
from experiment_tracker import log_experiment, promote_best, prune, update_readme


# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Cleaned source datasets (tracked in the repo).
EA_PATH = DATA_DIR / "player_stats_cleaned.csv"
TM_PATH = DATA_DIR / "transfermarkt_merged_players_with_valuation.csv"
FB_PATH = DATA_DIR / "Fbref_Final_Data.csv"

# Model outputs are written straight into data/ (what the dashboard reads).
OUTPUT_DIR = DATA_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_RUNS = 15

# Blend weights used in the expert layer.
EA_EXPERT_WEIGHT = 0.30
FB_EXPERT_WEIGHT = 0.25
TM_EXPERT_WEIGHT = 0.45


# ============================================================
# GENERAL HELPERS
# ============================================================
def normalize_name(name):
    if pd.isna(name):
        return np.nan
    name = str(name)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def safe_datetime(series):
    return pd.to_datetime(series, errors="coerce")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    pct_error = np.where(y_true != 0, (y_pred - y_true) / y_true, np.nan)
    abs_pct_error = np.abs(pct_error)

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Accuracy@10%": float(np.nanmean(abs_pct_error <= 0.10) * 100),
        "Accuracy@20%": float(np.nanmean(abs_pct_error <= 0.20) * 100),
        "Mean Percentage Error": float(np.nanmean(pct_error) * 100),
        "MAPE": float(np.nanmean(abs_pct_error) * 100),
    }


def get_existing_columns(df, cols):
    return [c for c in cols if c in df.columns]


def infer_position_group(pos):
    if pd.isna(pos):
        return "UNK"

    pos = str(pos).upper()

    if "GK" in pos:
        return "GK"

    if any(t in pos for t in ["CB", "LB", "RB", "LWB", "RWB", "DEF", "DF"]):
        return "DEF"

    if any(t in pos for t in ["DM", "CM", "AM", "LM", "RM", "MID", "MF", "CAM", "CDM"]):
        return "MID"

    if any(t in pos for t in ["ST", "CF", "LW", "RW", "FW", "WF", "ATT"]):
        return "FWD"

    return "UNK"


def add_position_group(df):
    df = df.copy()

    if "positions" in df.columns:
        df["position_group"] = df["positions"].apply(infer_position_group)
    elif "Pos" in df.columns:
        df["position_group"] = df["Pos"].apply(infer_position_group)
    elif "sub_position" in df.columns:
        df["position_group"] = df["sub_position"].apply(infer_position_group)
    elif "position" in df.columns:
        df["position_group"] = df["position"].apply(infer_position_group)
    else:
        df["position_group"] = "UNK"

    return df


def make_player_id(df):
    """Stable player identifier for multi-run aggregation (name_key + dob)."""
    df = df.copy()

    if "dob" in df.columns:
        dob_series = pd.to_datetime(df["dob"], errors="coerce").dt.strftime("%Y-%m-%d")
    elif "date_of_birth" in df.columns:
        dob_series = pd.to_datetime(df["date_of_birth"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        dob_series = pd.Series(["unknown_dob"] * len(df), index=df.index)

    if "name_key" in df.columns:
        name_series = df["name_key"].fillna("unknown_name").astype(str)
    elif "full_name" in df.columns:
        name_series = df["full_name"].apply(normalize_name).fillna("unknown_name")
    elif "name" in df.columns:
        name_series = df["name"].apply(normalize_name).fillna("unknown_name")
    elif "Player" in df.columns:
        name_series = df["Player"].apply(normalize_name).fillna("unknown_name")
    else:
        name_series = pd.Series(["unknown_name"] * len(df), index=df.index)

    df["player_id"] = name_series.astype(str) + "__" + dob_series.astype(str)
    return df


# ============================================================
# FEATURE NAME SANITISATION
# ============================================================
def clean_feature_name(name):
    name = str(name)
    name = name.replace("%", "_pct")
    name = name.replace("+", "_plus_")
    name = name.replace("/", "_per_")
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if name == "":
        name = "feature"
    return name


def clean_feature_columns(df):
    df = df.copy()

    new_cols = [clean_feature_name(c) for c in df.columns]

    counts = {}
    unique_cols = []
    for col in new_cols:
        if col not in counts:
            counts[col] = 0
            unique_cols.append(col)
        else:
            counts[col] += 1
            unique_cols.append(f"{col}_{counts[col]}")

    df.columns = unique_cols
    return df


# ============================================================
# FBREF FEATURE ENGINEERING
# ============================================================
def add_fbref_per90(df):
    df = df.copy()

    numeric_cols = [
        "90s", "Age", "Ast", "G+A", "G-PK", "G/Sh", "G/SoT", "Gls/90", "Int/90",
        "MP", "Min", "PK", "PKatt", "PKm", "Sh/90", "SoT%", "SoT/90",
        "Starts", "Subs", "TklW/90", "unSub", "Gls", "Int", "Sh", "SoT"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"Gls", "Ast"}.issubset(df.columns):
        df["goal_contrib"] = df["Gls"] + df["Ast"]

    if {"Gls", "Ast", "90s"}.issubset(df.columns):
        df["goal_contrib_per90"] = np.where(df["90s"] > 0, (df["Gls"] + df["Ast"]) / df["90s"], np.nan)

    if {"Starts", "MP"}.issubset(df.columns):
        df["start_ratio"] = np.where(df["MP"] > 0, df["Starts"] / df["MP"], np.nan)

    if {"Subs", "MP"}.issubset(df.columns):
        df["sub_ratio"] = np.where(df["MP"] > 0, df["Subs"] / df["MP"], np.nan)

    if {"Min", "MP"}.issubset(df.columns):
        df["minutes_per_match"] = np.where(df["MP"] > 0, df["Min"] / df["MP"], np.nan)

    return df


# ============================================================
# DATA PREPARATION
# ============================================================
def prepare_ea(ea):
    ea = ea.copy()

    if "full_name" in ea.columns:
        ea["name_key"] = ea["full_name"].fillna(ea.get("name")).apply(normalize_name)
    else:
        ea["name_key"] = ea["name"].apply(normalize_name)

    ea["dob"] = safe_datetime(ea["dob"])

    numeric_cols = [
        "height_cm", "weight_kg", "value", "wage", "preferred_foot", "weak_foot",
        "skill_moves", "release_clause", "overall_rating", "potential",
        "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
        "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve",
        "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
        "movement_acceleration", "movement_sprint_speed", "movement_agility",
        "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
        "power_stamina", "power_strength", "power_long_shots", "mentality_aggression",
        "mentality_interceptions", "mentality_vision", "mentality_penalties",
        "mentality_composure", "defending_defensive_awareness",
        "defending_standing_tackle", "defending_sliding_tackle",
        "goalkeeping_gk_diving", "goalkeeping_gk_handling", "goalkeeping_gk_kicking",
        "goalkeeping_gk_positioning", "goalkeeping_gk_reflexes", "mentality_attack_position"
    ]

    for col in numeric_cols:
        if col in ea.columns:
            ea[col] = pd.to_numeric(ea[col], errors="coerce")

    ref_date = pd.Timestamp("2025-01-01")
    ea["age_ea"] = (ref_date - ea["dob"]).dt.days / 365.25
    ea["age_ea_sq"] = ea["age_ea"] ** 2

    if "club_contract_valid_until" in ea.columns:
        ea["club_contract_valid_until"] = safe_datetime(ea["club_contract_valid_until"])
        ea["contract_years_left_ea"] = ((ea["club_contract_valid_until"] - ref_date).dt.days / 365.25).clip(lower=0)

    ea["is_left_footed"] = np.where(ea["preferred_foot"] == 1, 1, 0)

    ea["bmi_like"] = np.where(
        ea["height_cm"].notna() & ea["weight_kg"].notna() & (ea["height_cm"] > 0),
        ea["weight_kg"] / ((ea["height_cm"] / 100.0) ** 2),
        np.nan
    )

    if {"wage", "overall_rating"}.issubset(ea.columns):
        ea["wage_per_rating"] = ea["wage"] / ea["overall_rating"].replace(0, np.nan)

    ea = add_position_group(ea)
    ea = make_player_id(ea)
    return ea


def prepare_tm(tm):
    tm = tm.copy()

    tm["name_key"] = tm["name"].apply(normalize_name)
    tm["date_of_birth"] = safe_datetime(tm["date_of_birth"])

    if "contract_expiration_date" in tm.columns:
        tm["contract_expiration_date"] = safe_datetime(tm["contract_expiration_date"])

    if "date" in tm.columns:
        tm["date"] = safe_datetime(tm["date"])

    for col in ["market_value_in_eur_valuation", "age", "contract_years_left", "age_at_valuation", "is_free_agent"]:
        if col in tm.columns:
            tm[col] = pd.to_numeric(tm[col], errors="coerce")

    if {"name_key", "date_of_birth", "date"}.issubset(tm.columns):
        tm = tm.sort_values(["name_key", "date_of_birth", "date"], ascending=[True, True, False])
        tm = tm.drop_duplicates(["name_key", "date_of_birth"], keep="first").copy()

    if "age_at_valuation" in tm.columns:
        tm["age_at_valuation_sq"] = tm["age_at_valuation"] ** 2
        tm["abs_years_from_27"] = (tm["age_at_valuation"] - 27).abs()

    ref_date = pd.Timestamp("2025-01-01")
    if "contract_expiration_date" in tm.columns:
        tm["contract_years_left_from_date"] = ((tm["contract_expiration_date"] - ref_date).dt.days / 365.25).clip(lower=0)

    tm = add_position_group(tm)
    tm = make_player_id(tm)
    return tm


def prepare_fb(fb):
    fb = fb.copy()

    fb["name_key"] = fb["Player"].apply(normalize_name)
    fb = add_fbref_per90(fb)
    fb = add_position_group(fb)

    if {"name_key", "Min"}.issubset(fb.columns):
        fb = fb.sort_values(["name_key", "Min"], ascending=[True, False]).drop_duplicates("name_key", keep="first").copy()
    else:
        fb = fb.drop_duplicates("name_key", keep="first").copy()

    fb = make_player_id(fb)
    return fb


def add_target(df):
    """Target = average(EA value, TM value) when both exist."""
    df = df.copy()

    ea_val = pd.to_numeric(df["value"], errors="coerce")
    tm_val = pd.to_numeric(df["market_value_in_eur_valuation"], errors="coerce")

    df["target_value"] = np.where(
        ea_val.notna() & tm_val.notna(),
        (ea_val + tm_val) / 2.0,
        np.where(ea_val.notna(), ea_val, tm_val)
    )

    df = df[df["target_value"].notna() & (df["target_value"] > 0)].copy()
    df["log_target"] = np.log1p(df["target_value"])
    return df


# ============================================================
# FEATURE MATRIX HELPERS
# ============================================================
def make_design_matrices(train_df, test_df, feature_cols):
    if len(test_df) > 0:
        combined = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0)
        combined = pd.get_dummies(combined, dummy_na=True)
        x_train = combined.iloc[:len(train_df)].copy()
        x_test = combined.iloc[len(train_df):].copy()
    else:
        x_train = pd.get_dummies(train_df[feature_cols], dummy_na=True)
        x_test = pd.DataFrame(columns=x_train.columns)

    for col in x_train.columns:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")

    if len(x_test) > 0:
        for col in x_test.columns:
            x_test[col] = pd.to_numeric(x_test[col], errors="coerce")

    x_train = clean_feature_columns(x_train)

    if len(x_test) > 0:
        x_test = x_test.copy()
        x_test.columns = x_train.columns

    return x_train, x_test


def fit_imputer(x_train, x_test=None):
    imputer = SimpleImputer(strategy="median")

    x_train_i = pd.DataFrame(
        imputer.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index
    )

    if x_test is not None and len(x_test) > 0:
        x_test_i = pd.DataFrame(
            imputer.transform(x_test),
            columns=x_test.columns,
            index=x_test.index
        )
    else:
        x_test_i = pd.DataFrame(columns=x_train.columns)

    return x_train_i, x_test_i, imputer


# ============================================================
# MODEL BUILDERS
# ============================================================
def build_ea_model(random_state):
    if USE_LIGHTGBM:
        return LGBMRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=6, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=random_state, verbosity=-1
        )
    return ExtraTreesRegressor(
        n_estimators=500, max_depth=14, min_samples_leaf=2, random_state=random_state, n_jobs=-1
    )


def build_fb_model(random_state):
    if USE_XGBOOST:
        return XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, objective="reg:squarederror", random_state=random_state, n_jobs=-1
        )
    return ExtraTreesRegressor(
        n_estimators=500, max_depth=14, min_samples_leaf=2, random_state=random_state, n_jobs=-1
    )


def build_tm_model(random_state):
    if USE_CATBOOST:
        return CatBoostRegressor(
            iterations=700, learning_rate=0.03, depth=6, loss_function="RMSE",
            random_seed=random_state, verbose=0
        )
    return ExtraTreesRegressor(
        n_estimators=500, max_depth=14, min_samples_leaf=2, random_state=random_state, n_jobs=-1
    )


def build_boosted_meta_model(random_state):
    if USE_LIGHTGBM:
        return LGBMRegressor(
            n_estimators=700, learning_rate=0.025, max_depth=6, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=random_state, verbosity=-1
        )
    return ExtraTreesRegressor(
        n_estimators=600, max_depth=14, min_samples_leaf=2, random_state=random_state, n_jobs=-1
    )


def build_neural_meta_model(random_state):
    return MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
        alpha=1e-4, batch_size=128, learning_rate_init=0.001, max_iter=400,
        early_stopping=True, validation_fraction=0.15, random_state=random_state
    )


# ============================================================
# STACKING HELPERS
# ============================================================
def generate_oof_preds(train_df, test_df, feature_cols, target_col, model_builder, random_state, label):
    x_train, x_test = make_design_matrices(train_df, test_df, feature_cols)
    y_train = train_df[target_col].values

    oof = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df)) if len(test_df) > 0 else np.array([])

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train), start=1):
        x_tr = x_train.iloc[tr_idx]
        x_va = x_train.iloc[va_idx]
        y_tr = y_train[tr_idx]

        x_tr_i, x_va_i, imputer = fit_imputer(x_tr, x_va)
        model = model_builder(random_state + fold)
        model.fit(x_tr_i, y_tr)

        oof[va_idx] = model.predict(x_va_i)

        if len(test_df) > 0:
            x_test_i = pd.DataFrame(
                imputer.transform(x_test),
                columns=x_test.columns,
                index=x_test.index
            )
            test_preds += model.predict(x_test_i) / N_SPLITS

        print(f"[{label}] fold {fold}/{N_SPLITS} complete")

    x_train_i, x_test_i, final_imputer = fit_imputer(x_train, x_test if len(test_df) > 0 else None)
    final_model = model_builder(random_state)
    final_model.fit(x_train_i, y_train)

    return {
        "oof": oof,
        "test_preds": test_preds,
        "model": final_model,
        "imputer": final_imputer,
        "x_train": x_train_i,
        "x_test": x_test_i,
    }


def train_mlp_meta(train_meta, valid_meta, test_meta, feature_cols, target_col, random_state):
    x_train = pd.get_dummies(train_meta[feature_cols], dummy_na=True)
    x_valid = pd.get_dummies(valid_meta[feature_cols], dummy_na=True)
    x_test = pd.get_dummies(test_meta[feature_cols], dummy_na=True)

    all_cols = sorted(set(x_train.columns) | set(x_valid.columns) | set(x_test.columns))
    x_train = x_train.reindex(columns=all_cols, fill_value=0)
    x_valid = x_valid.reindex(columns=all_cols, fill_value=0)
    x_test = x_test.reindex(columns=all_cols, fill_value=0)

    x_train = clean_feature_columns(x_train).apply(pd.to_numeric, errors="coerce")
    x_valid = clean_feature_columns(x_valid).apply(pd.to_numeric, errors="coerce")
    x_test = clean_feature_columns(x_test).apply(pd.to_numeric, errors="coerce")

    # Ensure same columns after cleaning
    x_valid.columns = x_train.columns
    x_test.columns = x_train.columns

    imputer = SimpleImputer(strategy="median")
    x_train_i = imputer.fit_transform(x_train)
    x_valid_i = imputer.transform(x_valid)
    x_test_i = imputer.transform(x_test)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train_i)
    x_valid_s = scaler.transform(x_valid_i)
    x_test_s = scaler.transform(x_test_i)

    model = build_neural_meta_model(random_state)
    model.fit(x_train_s, train_meta[target_col].values)

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": list(x_train.columns),
        "valid_pred": model.predict(x_valid_s),
        "test_pred": model.predict(x_test_s)
    }


def add_blended_expert_features(df):
    df = df.copy()
    preds = ["ea_pred", "fb_pred", "tm_pred"]

    df["pred_mean"] = df[preds].mean(axis=1)
    df["pred_median"] = df[preds].median(axis=1)
    df["pred_weighted"] = (
        EA_EXPERT_WEIGHT * df["ea_pred"] +
        FB_EXPERT_WEIGHT * df["fb_pred"] +
        TM_EXPERT_WEIGHT * df["tm_pred"]
    )

    df["ea_fb_absdiff"] = np.abs(df["ea_pred"] - df["fb_pred"])
    df["ea_tm_absdiff"] = np.abs(df["ea_pred"] - df["tm_pred"])
    df["fb_tm_absdiff"] = np.abs(df["fb_pred"] - df["tm_pred"])

    df["pred_std"] = df[preds].std(axis=1)
    df["pred_max"] = df[preds].max(axis=1)
    df["pred_min"] = df[preds].min(axis=1)
    df["pred_range"] = df["pred_max"] - df["pred_min"]

    return df


# ============================================================
# FEATURE SETS
# ============================================================
def get_feature_sets(df):
    """Predictor sets per expert. Leakage columns (value,
    market_value_in_eur_valuation, release_clause) are deliberately excluded."""
    ea_features = get_existing_columns(df, [
        "position_group", "positions", "club_position", "club_league_name", "country_name",
        "height_cm", "weight_kg", "overall_rating", "potential",
        "preferred_foot", "is_left_footed", "weak_foot", "skill_moves",
        "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
        "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve",
        "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
        "movement_acceleration", "movement_sprint_speed", "movement_agility",
        "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
        "power_stamina", "power_strength", "power_long_shots", "mentality_aggression",
        "mentality_interceptions", "mentality_vision", "mentality_penalties",
        "mentality_composure", "mentality_attack_position",
        "defending_defensive_awareness", "defending_standing_tackle", "defending_sliding_tackle",
        "goalkeeping_gk_diving", "goalkeeping_gk_handling", "goalkeeping_gk_kicking",
        "goalkeeping_gk_positioning", "goalkeeping_gk_reflexes",
        "age_ea", "age_ea_sq", "contract_years_left_ea", "bmi_like"
    ])

    fb_features = get_existing_columns(df, [
        "position_group", "Pos", "Comp", "Nation", "Team",
        "Age", "90s", "MP", "Min", "Starts", "Subs", "unSub",
        "Ast", "G+A", "G-PK", "G/Sh", "G/SoT", "Gls/90", "Int/90", "Sh/90", "SoT%", "SoT/90", "TklW/90",
        "Gls", "Int", "Sh", "SoT",
        "goal_contrib", "goal_contrib_per90", "start_ratio", "sub_ratio", "minutes_per_match"
    ])

    tm_features = get_existing_columns(df, [
        "position_group", "sub_position", "country_of_citizenship", "current_club_name_valuation",
        "last_season", "age", "age_at_valuation", "age_at_valuation_sq",
        "contract_years_left", "contract_years_left_from_date", "abs_years_from_27", "is_free_agent"
    ])

    meta_features = get_existing_columns(df, [
        "position_group", "positions", "Pos", "sub_position",
        "club_league_name", "Comp", "country_name", "country_of_citizenship",
        "wage", "wage_per_rating",
        "overall_rating", "potential",
        "height_cm", "weight_kg", "bmi_like",
        "age_ea", "age_ea_sq", "age", "age_at_valuation", "age_at_valuation_sq", "abs_years_from_27",
        "contract_years_left_ea", "contract_years_left", "contract_years_left_from_date",
        "is_free_agent", "90s", "Min", "goal_contrib_per90", "Gls/90", "Int/90", "Sh/90", "SoT/90"
    ])

    return ea_features, fb_features, tm_features, meta_features


# ============================================================
# SINGLE RUN PIPELINE
# ============================================================
def run_single_pipeline(merged_df, run_number=1, random_state=42, save_shap=False):
    print("\n" + "=" * 70)
    print(f"RUN {run_number}")
    print("=" * 70)

    np.random.seed(random_state)

    train_df, test_df = train_test_split(merged_df, test_size=TEST_SIZE, random_state=random_state)

    ea_features, fb_features, tm_features, meta_context = get_feature_sets(merged_df)

    print("\nLayer 1 - Training EA expert...")
    ea_result = generate_oof_preds(
        train_df=train_df, test_df=test_df, feature_cols=ea_features, target_col="log_target",
        model_builder=build_ea_model, random_state=random_state, label=f"EA_EXPERT_RUN_{run_number}"
    )
    train_df["ea_pred"] = ea_result["oof"]
    test_df["ea_pred"] = ea_result["test_preds"]

    print("\nLayer 1 - Training FBref expert...")
    fb_result = generate_oof_preds(
        train_df=train_df, test_df=test_df, feature_cols=fb_features, target_col="log_target",
        model_builder=build_fb_model, random_state=random_state + 100, label=f"FB_EXPERT_RUN_{run_number}"
    )
    train_df["fb_pred"] = fb_result["oof"]
    test_df["fb_pred"] = fb_result["test_preds"]

    print("\nLayer 1 - Training TM expert...")
    tm_result = generate_oof_preds(
        train_df=train_df, test_df=test_df, feature_cols=tm_features, target_col="log_target",
        model_builder=build_tm_model, random_state=random_state + 200, label=f"TM_EXPERT_RUN_{run_number}"
    )
    train_df["tm_pred"] = tm_result["oof"]
    test_df["tm_pred"] = tm_result["test_preds"]

    train_df = add_blended_expert_features(train_df)
    test_df = add_blended_expert_features(test_df)

    blended_features = [
        "ea_pred", "fb_pred", "tm_pred",
        "pred_mean", "pred_median", "pred_weighted",
        "ea_fb_absdiff", "ea_tm_absdiff", "fb_tm_absdiff",
        "pred_std", "pred_max", "pred_min", "pred_range"
    ]

    test_df["has_fbref"] = test_df["Player"].notna().astype(int) if "Player" in test_df.columns else 0
    train_df["has_fbref"] = train_df["Player"].notna().astype(int) if "Player" in train_df.columns else 0

    meta_feature_cols = [c for c in blended_features + meta_context + ["has_fbref"] if c in train_df.columns]

    # Internal validation split for final blending
    meta_train, meta_valid = train_test_split(train_df, test_size=0.20, random_state=random_state)

    print("\nLayer 3A - Training boosted meta model...")
    x_meta_train, x_meta_valid = make_design_matrices(meta_train, meta_valid, meta_feature_cols)
    x_meta_train_i, x_meta_valid_i, _ = fit_imputer(x_meta_train, x_meta_valid)

    boosted_meta_model = build_boosted_meta_model(random_state + 300)
    boosted_meta_model.fit(x_meta_train_i, meta_train["log_target"].values)
    boosted_valid_pred = boosted_meta_model.predict(x_meta_valid_i)

    x_full_train, x_full_test = make_design_matrices(train_df, test_df, meta_feature_cols)
    x_full_train_i, x_full_test_i, _ = fit_imputer(x_full_train, x_full_test)

    boosted_meta_full = build_boosted_meta_model(random_state + 301)
    boosted_meta_full.fit(x_full_train_i, train_df["log_target"].values)
    boosted_test_pred = boosted_meta_full.predict(x_full_test_i)

    print("\nLayer 3B - Training neural meta model...")
    neural_result = train_mlp_meta(
        train_meta=meta_train, valid_meta=meta_valid, test_meta=test_df,
        feature_cols=meta_feature_cols, target_col="log_target", random_state=random_state + 400
    )
    neural_valid_pred = neural_result["valid_pred"]
    neural_test_pred = neural_result["test_pred"]

    # Final blend weights based on inverse validation RMSE
    y_valid = np.expm1(meta_valid["log_target"].values)
    boosted_valid_real = np.expm1(boosted_valid_pred)
    neural_valid_real = np.expm1(neural_valid_pred)

    boosted_rmse = np.sqrt(mean_squared_error(y_valid, boosted_valid_real))
    neural_rmse = np.sqrt(mean_squared_error(y_valid, neural_valid_real))

    boosted_weight = 1.0 / max(boosted_rmse, 1e-8)
    neural_weight = 1.0 / max(neural_rmse, 1e-8)
    total_weight = boosted_weight + neural_weight
    boosted_weight /= total_weight
    neural_weight /= total_weight

    final_test_pred_log = boosted_weight * boosted_test_pred + neural_weight * neural_test_pred

    y_test = np.expm1(test_df["log_target"].values)
    y_pred = np.expm1(final_test_pred_log)

    metrics = regression_metrics(y_test, y_pred)
    metrics["run"] = run_number
    metrics["boosted_meta_weight"] = float(boosted_weight)
    metrics["neural_meta_weight"] = float(neural_weight)

    print("\nMetrics for this run:")
    for k, v in metrics.items():
        if k != "run":
            print(f"{k}: {v:,.4f}")

    name_col = "full_name" if "full_name" in test_df.columns else "name"

    predictions = test_df[["player_id"]].copy()
    if name_col in test_df.columns:
        predictions["player_name"] = test_df[name_col].values
    else:
        predictions["player_name"] = test_df["name_key"].values

    predictions["position_group"] = test_df["position_group"].values if "position_group" in test_df.columns else "UNK"
    predictions["actual_value"] = y_test
    predictions["predicted_value"] = y_pred
    predictions["boosted_meta_pred"] = np.expm1(boosted_test_pred)
    predictions["neural_meta_pred"] = np.expm1(neural_test_pred)
    predictions["abs_pct_error"] = np.abs((predictions["predicted_value"] - predictions["actual_value"]) / predictions["actual_value"]) * 100
    predictions["run"] = run_number

    shap_df = None
    if save_shap and USE_SHAP:
        try:
            x_sample = x_full_train_i.sample(min(500, len(x_full_train_i)), random_state=random_state)
            explainer = shap.TreeExplainer(boosted_meta_full)
            shap_values = explainer.shap_values(x_sample)
            shap_df = pd.DataFrame({
                "feature": x_sample.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0)
            }).sort_values("mean_abs_shap", ascending=False)
            shap_df["run"] = run_number
        except Exception as e:
            print("SHAP error:", e)

    return {"metrics": metrics, "predictions": predictions, "shap_df": shap_df}


# ============================================================
# MULTI-RUN WRAPPER
# ============================================================
def run_multiple_experiments(merged_df, n_runs=15, base_random_state=42, save_shap=False):
    all_metrics = []
    all_predictions_long = []
    all_predictions_wide = None
    all_shap = []

    print("\n" + "#" * 70)
    print(f"STARTING MULTI-RUN EXPERIMENT: {n_runs} RUNS")
    print("#" * 70)

    for run in range(1, n_runs + 1):
        run_seed = base_random_state + run - 1

        result = run_single_pipeline(
            merged_df=merged_df.copy(), run_number=run, random_state=run_seed, save_shap=save_shap
        )

        all_metrics.append(result["metrics"])
        all_predictions_long.append(result["predictions"].copy())

        preds_wide = result["predictions"][["player_id", "player_name", "position_group", "actual_value", "predicted_value"]].copy()
        preds_wide = preds_wide.rename(columns={"predicted_value": f"pred_run_{run}"})

        if all_predictions_wide is None:
            all_predictions_wide = preds_wide
        else:
            all_predictions_wide = all_predictions_wide.merge(
                preds_wide[["player_id", f"pred_run_{run}"]], on="player_id", how="outer"
            )

        if result["shap_df"] is not None:
            all_shap.append(result["shap_df"])

    # Aggregate metrics
    metrics_df = pd.DataFrame(all_metrics)
    metric_cols = [c for c in metrics_df.columns if c != "run"]
    metrics_summary_rows = []
    for metric in metric_cols:
        metrics_summary_rows.append({
            "metric": metric,
            "mean": metrics_df[metric].mean(),
            "std": metrics_df[metric].std(),
            "min": metrics_df[metric].min(),
            "max": metrics_df[metric].max()
        })
    metrics_summary_df = pd.DataFrame(metrics_summary_rows)

    # Aggregate predictions
    pred_cols = [c for c in all_predictions_wide.columns if c.startswith("pred_run_")]
    all_predictions_wide["predicted_value_mean"] = all_predictions_wide[pred_cols].mean(axis=1)
    all_predictions_wide["predicted_value_std"] = all_predictions_wide[pred_cols].std(axis=1)
    all_predictions_wide["predicted_value_median"] = all_predictions_wide[pred_cols].median(axis=1)

    # Aggregate SHAP
    shap_summary_df = None
    if len(all_shap) > 0:
        shap_all_df = pd.concat(all_shap, ignore_index=True)
        shap_summary_df = (
            shap_all_df.groupby("feature", as_index=False)["mean_abs_shap"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "mean_abs_shap_mean", "std": "mean_abs_shap_std"})
            .sort_values("mean_abs_shap_mean", ascending=False)
        )

    # Save outputs into data/ (intermediate long/wide files are git-ignored)
    metrics_path = os.path.join(OUTPUT_DIR, "all_runs_metrics.csv")
    metrics_summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    predictions_long_path = os.path.join(OUTPUT_DIR, "all_runs_predictions_long.csv")
    predictions_wide_path = os.path.join(OUTPUT_DIR, "all_runs_predictions_wide.csv")
    final_predictions_path = os.path.join(OUTPUT_DIR, "final_player_values.csv")
    metrics_json_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")

    metrics_df.to_csv(metrics_path, index=False)
    metrics_summary_df.to_csv(metrics_summary_path, index=False)
    pd.concat(all_predictions_long, ignore_index=True).to_csv(predictions_long_path, index=False)
    all_predictions_wide.to_csv(predictions_wide_path, index=False)

    final_predictions = all_predictions_wide[[
        "player_id", "player_name", "position_group", "actual_value",
        "predicted_value_mean", "predicted_value_median", "predicted_value_std"
    ]].copy()
    final_predictions.to_csv(final_predictions_path, index=False)

    metrics_summary_json = {}
    for _, row in metrics_summary_df.iterrows():
        metrics_summary_json[row["metric"]] = {
            "mean": float(row["mean"]), "std": float(row["std"]),
            "min": float(row["min"]), "max": float(row["max"])
        }
    save_json(metrics_summary_json, metrics_json_path)

    if shap_summary_df is not None:
        shap_summary_df.to_csv(os.path.join(OUTPUT_DIR, "shap_summary.csv"), index=False)

    print("\n" + "#" * 70)
    print("MULTI-RUN SUMMARY")
    print("#" * 70)
    print(metrics_summary_df.to_string(index=False))

    return {
        "metrics_df": metrics_df,
        "metrics_summary_df": metrics_summary_df,
        "predictions_wide": all_predictions_wide,
        "final_predictions": final_predictions,
        "shap_summary_df": shap_summary_df
    }


# ============================================================
# MAIN DATA BUILD
# ============================================================
def build_merged_dataset():
    print("Loading datasets...")
    ea = pd.read_csv(EA_PATH)
    tm = pd.read_csv(TM_PATH)
    fb = pd.read_csv(FB_PATH)

    print("Raw shapes:", "EA:", ea.shape, "TM:", tm.shape, "FB:", fb.shape)

    print("\nPreparing datasets...")
    ea = prepare_ea(ea)
    tm = prepare_tm(tm)
    fb = prepare_fb(fb)

    print("\nMerging EA and TM on normalized name + dob...")
    merged = ea.merge(
        tm, left_on=["name_key", "dob"], right_on=["name_key", "date_of_birth"],
        how="inner", suffixes=("_ea", "_tm")
    )

    print("Joining FBref on normalized name...")
    merged = merged.merge(fb, on="name_key", how="left", suffixes=("", "_fb"))

    merged = add_position_group(merged)
    merged = add_target(merged)
    merged = make_player_id(merged)
    merged["has_fbref"] = merged["Player"].notna().astype(int) if "Player" in merged.columns else 0

    print("Final merged shape:", merged.shape, "| Unique players:", merged["player_id"].nunique())
    return merged


# ============================================================
# PIPELINE ENTRY POINT
# ============================================================
def train_model() -> None:
    """Retrain the model and write the output files into data/."""
    merged_df = build_merged_dataset()
    run_multiple_experiments(
        merged_df=merged_df, n_runs=N_RUNS, base_random_state=BASE_RANDOM_STATE, save_shap=True
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
