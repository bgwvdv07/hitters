from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUTPUT_DIR = Path("output")
TARGET_COL_1 = "had_hit_1"
TARGET_COL_2 = "had_hit_2"
TARGET_COL_3 = "had_run_1"

logger = logging.getLogger("train_hit_model")


def setup_logging(verbosity):
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def write_json(path: Path, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_pickle(path: Path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_history(history_path: Path) -> pd.DataFrame:
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    df = pd.read_csv(history_path)
    if df.empty:
        raise ValueError(f"History file is empty: {history_path}")

    # Require at least one of the two targets; we'll derive the other if needed
    if "had_hit_1" not in df.columns and "had_hit" not in df.columns:
        raise ValueError(
            "Missing required target column: need 'had_hit' or 'had_hit_1' in history."
        )

    # Derive had_hit_1 and had_hit_2 if not present
    if "had_hit_1" not in df.columns:
        df["had_hit_1"] = df["had_hit"].astype(int)

    if "had_hit_2" not in df.columns:
        if "hit_count" in df.columns:
            df["had_hit_2"] = (df["hit_count"] >= 2).astype(int)
        else:
            # Fallback: assume multi-hit is rarer; use 0 as placeholder
            logger.warning(
                "hit_count not found; cannot derive had_hit_2 accurately. "
                "Training will use had_hit_1 only for both targets."
            )
            df["had_hit_2"] = df["had_hit_1"].astype(int)

    if "had_run_1" not in df.columns:
        if "runs_scored" in df.columns:
            df["had_run_1"] = (pd.to_numeric(df["runs_scored"], errors="coerce").fillna(0) >= 1).astype(int)
        else:
            logger.warning(
                "runs_scored not found in history; cannot train a run-probability model. "
                "Re-run update_training_history.py / build_graded_history.py on graded picks "
                "that include runs_scored, then retrain."
            )
            df["had_run_1"] = None

    return df


def infer_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {
        "date",
        "graded_at",
        "trained_at",
        "training_row_created_at",
        "batter",
        "matchup",
        "batting_team",
        "opp_team",
        "opp_pitcher",
        "batter_api_name",
        "team_name_api",
        "result_label",
        "qa_error",
        # identifiers, not predictive signal
        "game_pk",
        "batter_id",
        # targets
        "had_hit",
        "had_hit_1",
        "had_hit_2",
        "had_run_1",
        # post-game outcome columns -- these are only known AFTER the game and
        # directly determine the targets above (e.g. hit_count>=2 == had_hit_2).
        # Leaving these in as features is target leakage: the model would just
        # learn to read the answer instead of predicting it.
        "hit_count",
        "single_count",
        "double_count",
        "triple_count",
        "home_run_count",
        "xbh_count",
        "at_bats",
        "plate_appearances",
        "runs_scored",
        "rbi",
        "walks",
        "strikeouts",
    }

    feature_cols = [c for c in df.columns if c not in excluded]

    categorical_features = [
        c for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]

    numeric_features = [c for c in feature_cols if c not in categorical_features]

    return numeric_features, categorical_features


def prepare_training_frame(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    use_cols = numeric_features + categorical_features + [target_col]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")

    model_df = df[use_cols].copy()
    model_df = model_df.dropna(subset=[target_col]).copy()
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col]).copy()
    model_df[target_col] = model_df[target_col].astype(int)

    X = model_df[numeric_features + categorical_features].copy()
    y = model_df[target_col].copy()

    return X, y


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "log_loss": float(log_loss(y_test, probs)),
        "brier_score": float(brier_score_loss(y_test, probs)),
    }

    unique_classes = sorted(pd.Series(y_test).dropna().unique().tolist())
    if len(unique_classes) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
    else:
        metrics["roc_auc"] = None

    pred_df = X_test.copy()
    pred_df["actual"] = y_test.values
    pred_df["pred_hit_prob"] = probs
    pred_df["pred_label"] = preds

    return metrics, pred_df


def export_feature_coefficients(
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
    out_name: str,
):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)

    out_path = OUTPUT_DIR / out_name
    coef_df.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


def score_full_history(
    model: Pipeline,
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
    out_name: str,
):
    needed = [c for c in numeric_features + categorical_features if c in df.columns]
    scored = df.copy()
    X_all = scored[needed].copy()

    probs = model.predict_proba(X_all)[:, 1]
    scored[f"pred_{target_col}"] = probs
    scored[f"pred_{target_col}_label"] = (probs >= 0.5).astype(int)

    out_path = OUTPUT_DIR / out_name
    scored.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)


def score_today_file(
    model: Pipeline,
    today_path: Path,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
):
    if not today_path.exists():
        logger.warning("Today file not found, skipping scoring: %s", today_path)
        return None

    today_df = pd.read_csv(today_path)
    feature_cols = numeric_features + categorical_features
    missing = [c for c in feature_cols if c not in today_df.columns]
    if missing:
        raise ValueError(f"Today file missing required feature columns: {missing}")

    probs = model.predict_proba(today_df[feature_cols])[:, 1]
    scored_today = today_df.copy()
    scored_today[f"pred_{target_col}"] = probs
    scored_today[f"pred_{target_col}_label"] = (probs >= 0.5).astype(int)

    if f"pred_{target_col}" in scored_today.columns:
        sort_cols = [f"pred_{target_col}"]
        ascending = [False]
        if "matchup" in scored_today.columns and "order" in scored_today.columns:
            sort_cols = ["matchup", f"pred_{target_col}", "order"]
            ascending = [True, False, True]
        scored_today = scored_today.sort_values(sort_cols, ascending=ascending)

    out_path = OUTPUT_DIR / f"today_model_scored_{target_col}_{today_path.stem}.csv"
    scored_today.to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)

    return scored_today


def train_target(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_col: str,
    args,
):
    logger.info("Training target: %s", target_col)

    X, y = prepare_training_frame(df, numeric_features, categorical_features, target_col)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    trained_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    model_bundle = {
        "pipeline": model,
        "feature_columns": numeric_features + categorical_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target_col,
        "class_labels": model.classes_.tolist() if hasattr(model, "classes_") else None,
        "trained_at": trained_at,
    }

    bundle_name = f"{target_col}_model_bundle.pkl"
    write_pickle(OUTPUT_DIR / bundle_name, model_bundle)
    logger.info("Wrote %s", OUTPUT_DIR / bundle_name)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model pipeline does not support predict_proba().")

    metrics, test_predictions = evaluate_model(model, X_test, y_test)
    test_pred_path = OUTPUT_DIR / f"test_predictions_{target_col}.csv"
    test_predictions.to_csv(test_pred_path, index=False)
    logger.info("Wrote %s", test_pred_path)

    export_feature_coefficients(
        model,
        numeric_features,
        categorical_features,
        out_name=f"feature_coefficients_{target_col}.csv",
    )

    score_full_history(
        model,
        df,
        numeric_features,
        categorical_features,
        target_col,
        out_name=f"full_history_scored_{target_col}.csv",
    )

    if args.today_file:
        score_today_file(
            model=model,
            today_path=Path(args.today_file),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_col=target_col,
        )

    model_info = {
        "history_file": str(Path(args.history_file)),
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": numeric_features + categorical_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target_col,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "metrics": metrics,
        "trained_at": trained_at,
    }

    write_json(OUTPUT_DIR / f"model_info_{target_col}.json", model_info)
    logger.info("Wrote %s", OUTPUT_DIR / f"model_info_{target_col}.json")

    return model_info



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train MLB hit-probability models (1+ hit and 2+ hits) from graded history "
            "and export bundles/artifacts."
        )
    )
    parser.add_argument(
        "--history-file",
        default=str(OUTPUT_DIR / "graded_history.csv"),
        help="CSV containing historical graded picks with had_hit / hit_count.",
    )
    parser.add_argument(
        "--today-file",
        default=None,
        help="Optional CSV of today's candidates to score after training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of rows reserved for test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Minimum required training rows.",
    )
    parser.add_argument(
        "--numeric-features",
        nargs="*",
        default=None,
        help="Optional explicit numeric feature list.",
    )
    parser.add_argument(
        "--categorical-features",
        nargs="*",
        default=None,
        help="Optional explicit categorical feature list.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="increase terminal verbosity (-v for progress, -vv for debug)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history_path = Path(args.history_file)
    df = load_history(history_path)
    

    if len(df) < args.min_rows:
        raise ValueError(
            f"Not enough rows to train model: {len(df)} found, {args.min_rows} required."
        )

    if args.numeric_features is not None or args.categorical_features is not None:
        numeric_features = args.numeric_features or []
        categorical_features = args.categorical_features or []
    else:
        numeric_features, categorical_features = infer_feature_sets(df)

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns found for training.")

    logger.info("Rows loaded: %s", len(df))
    logger.info("Numeric features: %s", numeric_features)
    logger.info("Categorical features: %s", categorical_features)

    # Train 1+ hit model
    info_1 = train_target(
        df,
        numeric_features,
        categorical_features,
        TARGET_COL_1,
        args,
    )

    # Train 2+ hit model
    info_2 = train_target(
        df,
        numeric_features,
        categorical_features,
        TARGET_COL_2,
        args,
    )

    # Train run-scored model, if we have runs_scored-derived labels to learn from
    info_3 = None
    targets_trained = [TARGET_COL_1, TARGET_COL_2]
    if df[TARGET_COL_3].notna().any():
        info_3 = train_target(
            df,
            numeric_features,
            categorical_features,
            TARGET_COL_3,
            args,
        )
        targets_trained.append(TARGET_COL_3)
    else:
        logger.warning(
            "Skipping %s model: no runs_scored data in history yet.", TARGET_COL_3
        )

    # Summary
    summary = {
        "history_file": str(history_path),
        "rows_used": len(df),
        "targets": targets_trained,
        "model_info_1": info_1,
        "model_info_2": info_2,
        "model_info_3": info_3,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    write_json(OUTPUT_DIR / "model_summary.json", summary)
    logger.info("Wrote %s", OUTPUT_DIR / "model_summary.json")


if __name__ == "__main__":
    main()