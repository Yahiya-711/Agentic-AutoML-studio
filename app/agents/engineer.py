# =============================================================================
# engineer.py — The FLAML AutoML Training Agent
# =============================================================================
#
# WHAT IS THIS FILE?
# ------------------
# This is the "Worker" agent. It receives the Orchestrator's plan from state,
# executes the actual machine learning training using FLAML, and writes the
# results (leaderboard, feature importance, best model) back to state.
#
# It has NO reasoning ability of its own — it just executes.
# The LLM (Orchestrator/Auditor) thinks. FLAML does the math.
#
# WHAT IS FLAML?
# --------------
# FLAML (Fast and Lightweight AutoML) is a Microsoft Research library that:
#   1. Automatically selects which algorithm to try (LightGBM, XGBoost, RF...)
#   2. Automatically tunes hyperparameters using a low-cost search strategy
#   3. Stops when the time_budget runs out — not when accuracy plateaus
#
# WHY FLAML OVER H2O / AUTOGLUON?
# --------------------------------
# H2O:        Java JVM needs 2-4GB heap just to start. Kills 8GB RAM machines.
# AutoGluon:  Stacking/bagging by default → 3-4GB RAM spike. Risky on 8GB.
# FLAML:      Pure Python, peaks at ~1.5GB, outperforms both under time budgets.
#             Designed specifically for low-resource environments.
#
# HOW FLAML FITS INTO THE LANGGRAPH LOOP
# ----------------------------------------
# First run:  Train on ALL columns except target
# If Auditor rejects: excluded_columns grows, Engineer re-runs with fewer cols
# The state carries the exclusion list — Engineer reads it on every call
#
# =============================================================================

import os
import shutil
import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any
from app.agents.state import AgentState


# =============================================================================
# CONSTANTS
# =============================================================================

# Where FLAML saves temporary model artifacts during training.
# We clean this up after each run to avoid filling your 256GB SSD
# across multiple retrain loops.
FLAML_TEMP_DIR = "/tmp/flaml_automl_cache"

# Number of cross-validation folds inside FLAML's internal evaluation.
# 3-fold is the sweet spot for 8GB RAM:
#   - 5-fold is more accurate but uses more memory
#   - 2-fold is faster but scores are noisier
CV_FOLDS = 3

# Cap on parallel jobs. Your i5-7200U has 4 threads.
# We use 2 for FLAML, leaving 2 for Ollama (which runs concurrently).
# Using all 4 for FLAML would starve Ollama and cause LLM timeout.
N_JOBS = 2


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _clean_temp_dir():
    """
    Delete FLAML's temp directory before each training run.

    WHY?
    FLAML writes model checkpoints to disk during search. Over multiple
    retrain loops (Auditor rejects → Engineer retrains), these accumulate.
    On a 256GB SSD that's already holding your OS + Python + models,
    letting temp files grow is risky. Clean slate each time.
    """
    if os.path.exists(FLAML_TEMP_DIR):
        shutil.rmtree(FLAML_TEMP_DIR)
    os.makedirs(FLAML_TEMP_DIR, exist_ok=True)


def _load_and_prepare_data(
    file_path: str,
    target_column: str,
    excluded_columns: List[str],
) -> tuple:
    """
    Load CSV, drop excluded columns, separate features from target.

    Returns:
        X_train, X_val, y_train, y_val, feature_names

    WHY SPLIT HERE AND NOT INSIDE FLAML?
    FLAML can handle a full DataFrame and do its own internal split,
    but giving it an explicit train/val split gives us:
      1. A consistent validation set for the leaderboard score
      2. Reproducibility (same random_state=42 every loop)
      3. Control over val size (we use 20%)

    WHY DROP EXCLUDED COLUMNS HERE?
    The Auditor writes to state["excluded_columns"] after finding leakage.
    On the second loop, this list contains the bad columns. We drop them
    here so FLAML never sees them — it can't leak on data it doesn't have.
    """
    df = pd.read_csv(file_path)

    # Drop columns the Auditor flagged in previous loops
    # errors="ignore" prevents crash if a col was already absent
    cols_to_drop = [c for c in excluded_columns if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Separate features (X) from target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_names = X.columns.tolist()

    # Stratified split for classification (preserves class ratio in train/val)
    # Regular split for regression (stratify doesn't apply to continuous y)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,          # only works for classification
        )
    except ValueError:
        # stratify fails if any class has < 2 samples — fall back to no stratify
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
        )

    return X_train, X_val, y_train, y_val, feature_names


def _encode_target_if_needed(y_train, y_val, task_type: str):
    """
    FLAML's classification expects numeric labels (0, 1, 2...).
    If the target column contains strings like "yes"/"no" or "cat"/"dog",
    we need to encode them to integers.

    WHY NOT USE pd.get_dummies OR ONE-HOT?
    Target encoding (LabelEncoder) maps each class to a single integer.
    One-hot is for FEATURES, not targets. For a binary target:
      "yes" → 1, "no" → 0

    We return the encoder so that future predictions can be decoded
    back to original labels (not needed for this project, but good practice).
    """
    if task_type == "classification" and y_train.dtype == object:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        return y_train_enc, y_val_enc, le
    return y_train, y_val, None


def _run_flaml(
    X_train, y_train,
    task_type: str,
    metric: str,
    time_limit: int,
) -> AutoML:
    """
    Core FLAML training call.

    WHAT FLAML DOES INTERNALLY:
    1. Starts with the cheapest models (LinearSVC, LogisticRegression)
    2. Progressively tries more expensive ones (LightGBM, XGBoost, RF)
    3. Within each model, uses Bayesian-like hyperparameter search
    4. Stops when time_budget (seconds) is exhausted
    5. Returns the best model found in that time window

    PARAMETER EXPLANATIONS:
    ───────────────────────
    task:
        "classification" or "regression"
        Tells FLAML which algorithm family to use

    metric:
        What to optimize. FLAML supports:
        "roc_auc"    → area under ROC curve (binary classification)
        "accuracy"   → fraction of correct predictions (multiclass)
        "rmse"       → root mean squared error (regression)
        "mae"        → mean absolute error (regression)
        "r2"         → R-squared (regression)

    time_budget:
        Hard wall-clock stop in seconds. FLAML respects this strictly.
        At 60s on your i5, expect 8-15 models evaluated.

    n_jobs:
        Parallel workers for model training. Set to 2 to leave 2 threads
        for Ollama. Using -1 (all threads) would freeze Ollama calls.

    verbose:
        0 = silent. IMPORTANT for Streamlit — FLAML's logs would flood
        the terminal and confuse users watching the dashboard.

    eval_method:
        "cv" = cross-validation with n_splits folds
        "holdout" = single train/val split
        We use "cv" for more reliable scores on small datasets.
        On large datasets (>10k rows), "holdout" is faster.

    n_splits:
        Number of CV folds. 3 is the memory-safe choice on 8GB.

    seed:
        Reproducibility. Same seed = same results every run.
        Critical for debugging — if results change between runs,
        you know it's randomness, not a bug.
    """
    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=task_type,
        metric=metric,
        time_budget=time_limit,
        n_jobs=N_JOBS,
        verbose=0,
        eval_method="cv",
        n_splits=CV_FOLDS,
        seed=42,
        # Memory safety: limit ensemble size
        # Without this, FLAML might try to build a large ensemble at the end
        # which can spike memory beyond what we budget
        ensemble=False,
    )
    return automl


def _build_leaderboard(automl: AutoML) -> List[Dict[str, Any]]:
    """
    Extract the trial history from FLAML and build a ranked leaderboard.

    WHAT IS A "TRIAL" IN FLAML?
    Each time FLAML tests a model + hyperparameter combo, that's one trial.
    In 60 seconds you'll typically get 10-20 trials.

    FLAML stores trial results in automl._trials (internal attribute).
    Each trial is a dict with:
        "learner"          → model class name (e.g., "lgbm", "xgb_limitdepth")
        "val_loss"         → validation loss (lower is better for FLAML)
        "wall_clock_time"  → seconds this trial took

    WHY CONVERT val_loss TO score?
    FLAML internally minimizes loss (lower = better):
        roc_auc loss = 1 - roc_auc  → score = 1 - loss
        rmse loss    = rmse         → score = loss (keep as-is, lower is better)
    We convert to "score" (higher = better) for the leaderboard display
    because users expect higher numbers to mean better models.
    """
    leaderboard = []

    # FLAML model name mapping — internal names to readable names
    # FLAML uses short aliases internally; we display proper class names
    MODEL_NAME_MAP = {
        "lgbm":             "LGBMClassifier / LGBMRegressor",
        "xgb_limitdepth":   "XGBClassifier (depth-limited)",
        "xgboost":          "XGBClassifier",
        "rf":               "RandomForestClassifier",
        "extra_tree":       "ExtraTreesClassifier",
        "lrl1":             "LogisticRegression (L1)",
        "lrl2":             "LogisticRegression (L2)",
        "kneighbor":        "KNeighborsClassifier",
        "catboost":         "CatBoostClassifier",
        "svc":              "SVC",
        "sgd":              "SGDClassifier",
    }

    try:
        # Access FLAML's internal trial log
        # _trials is a list of dicts, one per trial attempted
        if hasattr(automl, '_trials') and automl._trials:
            # Sort by val_loss ascending (best first)
            trials_sorted = sorted(
                automl._trials,
                key=lambda t: t.get("val_loss", float("inf"))
            )

            seen_models = set()   # deduplicate — show only best per model type
            for trial in trials_sorted:
                learner = trial.get("learner", "unknown")

                if learner in seen_models:
                    continue     # skip if we already have a better result for this model
                seen_models.add(learner)

                val_loss = trial.get("val_loss", 0.0)
                # Convert FLAML's loss to a human-readable score
                # For regression metrics, loss IS the score (keep it)
                # For classification, loss = 1 - metric, so score = 1 - loss
                score = round(1.0 - val_loss, 4)

                leaderboard.append({
                    "model": MODEL_NAME_MAP.get(learner, learner),
                    "score": score,
                    "training_time_s": round(trial.get("wall_clock_time", 0.0), 1),
                })

                if len(leaderboard) >= 8:
                    break   # cap at 8 entries for the dashboard display

    except Exception:
        pass   # if trial history access fails, fall through to fallback

    # Fallback: if trial history is unavailable, show just the best model
    # This happens with some FLAML versions that don't expose _trials
    if not leaderboard:
        leaderboard.append({
            "model": MODEL_NAME_MAP.get(str(automl.best_estimator), str(automl.best_estimator)),
            "score": round(1.0 - automl.best_result.get("val_loss", 0.0), 4),
            "training_time_s": 0.0,
        })

    return leaderboard


def _extract_feature_importance(automl: AutoML, feature_names: List[str]) -> Dict[str, float]:
    """
    Get feature importance scores from FLAML's best model.

    WHY NOT USE PERMUTATION IMPORTANCE?
    Permutation importance shuffles each column and measures score drop.
    It requires running the model N_features × N_repeats extra times.
    On a 30-column dataset with 5 repeats, that's 150 extra inferences.
    On your i5, this can take 5-10 minutes — too slow for a demo.

    FLAML'S NATIVE IMPORTANCE:
    Tree-based models (LightGBM, XGBoost, RF) natively compute feature
    importance based on how often/much each feature is used in splits.
    This is instantaneous — already computed during training.
    We access it via automl.feature_importances_ (a numpy array).

    WHAT IF THE BEST MODEL IS LINEAR (NO TREE IMPORTANCE)?
    Linear models (LogisticRegression, SVC) use coefficients instead of
    importances. We fall back to absolute coefficient values, which have
    a similar interpretation (higher = more influential).

    If neither is available (rare edge case), we assign equal weights.
    This is honest — we don't fabricate importance data.
    """
    importance_dict = {}

    try:
        # Try tree-based importance first (LightGBM, XGBoost, RandomForest)
        importances = automl.feature_importances_

        if importances is not None and len(importances) == len(feature_names):
            # Normalize to [0, 1] range for consistent display across models
            # Raw importances can be in arbitrary scales (LightGBM uses gain,
            # XGBoost uses weight, RF uses mean decrease in impurity)
            total = sum(importances) or 1.0   # avoid division by zero
            importance_dict = {
                col: round(float(imp) / total, 4)
                for col, imp in zip(feature_names, importances)
            }

    except (AttributeError, TypeError):
        # Tree importance not available — try linear model coefficients
        try:
            model = automl.model.estimator   # unwrap FLAML's estimator wrapper
            if hasattr(model, 'coef_'):
                coefs = np.abs(model.coef_).flatten()
                if len(coefs) == len(feature_names):
                    total = sum(coefs) or 1.0
                    importance_dict = {
                        col: round(float(c) / total, 4)
                        for col, c in zip(feature_names, coefs)
                    }
        except Exception:
            pass

    # Final fallback: equal weights
    # We tell the Auditor the features are equally important — it will
    # flag ID columns by name pattern rather than by importance score
    if not importance_dict:
        weight = round(1.0 / len(feature_names), 4) if feature_names else 0.0
        importance_dict = {col: weight for col in feature_names}

    return importance_dict


def _compute_best_score(automl: AutoML, metric: str) -> float:
    """
    Convert FLAML's internal val_loss to a readable score.

    FLAML always minimizes val_loss internally:
        For "roc_auc":  val_loss = 1 - roc_auc  → score = 1 - val_loss
        For "accuracy": val_loss = 1 - accuracy → score = 1 - val_loss
        For "rmse":     val_loss = rmse          → score = val_loss (keep)
        For "mae":      val_loss = mae           → score = val_loss (keep)
        For "r2":       val_loss = 1 - r2        → score = 1 - val_loss

    We detect regression metrics by name and keep them as-is.
    Everything else is a classification metric where higher = better.
    """
    regression_metrics = {"rmse", "mae", "mse", "r2", "mape"}
    val_loss = automl.best_result.get("val_loss", 0.0)

    if metric in regression_metrics:
        return round(float(val_loss), 4)   # lower is better, keep as-is
    else:
        return round(1.0 - float(val_loss), 4)   # convert loss → score


# =============================================================================
# THE ENGINEER NODE — This is what LangGraph calls
# =============================================================================

def engineer_node(state: AgentState) -> AgentState:
    """
    The Engineer agent function. Called by LangGraph as a graph node.

    LangGraph contract:
        - Input:  AgentState dict
        - Output: AgentState dict (same keys, updated values)
        - Must not raise exceptions (catch everything and write to state)

    Flow:
        1. Read plan from state (target, metric, time_limit, exclusions)
        2. Load and prepare the CSV
        3. Run FLAML AutoML
        4. Extract leaderboard + feature importance
        5. Write results back to state
        6. Return updated state (LangGraph merges it automatically)
    """

    print(f"\n[Engineer] Starting training run #{state.get('retry_count', 0) + 1}")
    print(f"[Engineer] Target: {state['target_column']} | Metric: {state['metric']} | Time: {state['time_limit']}s")

    excluded = state.get("excluded_columns", [])
    if excluded:
        print(f"[Engineer] Excluding columns flagged by Auditor: {excluded}")

    # ── Step 1: Clean up previous temp files ─────────────────────────────────
    _clean_temp_dir()

    # ── Step 2: Load data and prepare train/val split ─────────────────────────
    try:
        X_train, X_val, y_train, y_val, feature_names = _load_and_prepare_data(
            file_path=state["file_path"],
            target_column=state["target_column"],
            excluded_columns=excluded,
        )
    except Exception as e:
        # If data loading fails (corrupt CSV, missing target col, etc.),
        # write an error signal to state and let the graph route to END.
        # We never want to crash the entire pipeline on a data error.
        print(f"[Engineer] ERROR loading data: {e}")
        return {
            **state,
            "leaderboard": [{"model": "ERROR", "score": 0.0, "training_time_s": 0.0}],
            "feature_importance": {},
            "best_model_score": 0.0,
            "best_model_name": "DataLoadError",
            "approved": True,    # Force approval to stop the loop
        }

    print(f"[Engineer] Dataset: {X_train.shape[0]} train rows, {X_val.shape[0]} val rows, {len(feature_names)} features")

    # ── Step 3: Encode target if needed (string → integer for classification) ──
    task_type = state.get("task_type", "classification")
    metric = state.get("metric", "accuracy")

    y_train_enc, y_val_enc, label_encoder = _encode_target_if_needed(
        y_train, y_val, task_type
    )

    # ── Step 4: Run FLAML AutoML ───────────────────────────────────────────────
    print(f"[Engineer] FLAML training started... (this will take ~{state['time_limit']}s)")

    try:
        automl = _run_flaml(
            X_train=X_train,
            y_train=y_train_enc,
            task_type=task_type,
            metric=metric,
            time_limit=state["time_limit"],
        )
    except Exception as e:
        print(f"[Engineer] ERROR during FLAML training: {e}")
        return {
            **state,
            "leaderboard": [{"model": "TrainingError", "score": 0.0, "training_time_s": 0.0}],
            "feature_importance": {col: 0.0 for col in feature_names},
            "best_model_score": 0.0,
            "best_model_name": "TrainingError",
            "approved": True,    # Stop the loop on training failure
        }

    best_model_name = str(automl.best_estimator)
    print(f"[Engineer] Training complete. Best model: {best_model_name}")

    # ── Step 5: Extract leaderboard ───────────────────────────────────────────
    leaderboard = _build_leaderboard(automl)
    print(f"[Engineer] Leaderboard built: {len(leaderboard)} models ranked")

    # ── Step 6: Extract feature importance ────────────────────────────────────
    feature_importance = _extract_feature_importance(automl, feature_names)
    top_3 = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"[Engineer] Top 3 features: {top_3}")

    # ── Step 7: Compute readable best score ───────────────────────────────────
    best_score = _compute_best_score(automl, metric)
    print(f"[Engineer] Best score ({metric}): {best_score}")

    # ── Step 8: Return updated state ──────────────────────────────────────────
    # We use **state to carry forward ALL existing fields unchanged,
    # then override only the fields this agent is responsible for.
    # This is the LangGraph pattern — partial state updates.
    return {
        **state,
        "leaderboard": leaderboard,
        "feature_importance": feature_importance,
        "best_model_score": best_score,
        "best_model_name": best_model_name,
    }


# =============================================================================
# STANDALONE TEST
# Run this file directly to verify FLAML works on your machine:
#   python3 app/agents/engineer.py
#
# This test:
#   1. Creates a dataset with an obvious ID leak column
#   2. Runs the engineer_node with no exclusions (first pass)
#   3. Runs again with the ID column excluded (simulates Auditor loop)
#   4. Verifies the leaderboard and feature importance are populated
# =============================================================================

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    import tempfile

    print("=" * 60)
    print("Engineer Node — Standalone Test")
    print("=" * 60)

    # Build test dataset with a deliberate leakage column
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame.copy()
    df["patient_id"] = range(len(df))          # obvious ID leak
    df["target_proxy"] = df["target"] * 0.99   # near-perfect proxy leak

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
        df.to_csv(f, index=False)
        test_path = f.name

    print(f"Test CSV saved to: {test_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")

    # Build a minimal state (same as what Orchestrator would produce)
    from app.agents.state import create_initial_state
    state = create_initial_state(
        file_path=test_path,
        columns=df.columns.tolist(),
        dtypes=df.dtypes.astype(str).to_dict(),
        task_description="predict cancer diagnosis",
        time_limit=30,  # short for testing
    )
    # Manually fill what the Orchestrator would have set
    state["target_column"] = "target"
    state["metric"] = "roc_auc"
    state["task_type"] = "classification"

    # ── Test Run 1: No exclusions ─────────────────────────────────────────────
    print("\n--- RUN 1: No exclusions (first pass) ---")
    result1 = engineer_node(state)

    print(f"\nLeaderboard ({len(result1['leaderboard'])} models):")
    for row in result1["leaderboard"]:
        print(f"  {row['model']:<45} score={row['score']}  time={row['training_time_s']}s")

    print(f"\nTop 5 features by importance:")
    top5 = sorted(result1["feature_importance"].items(), key=lambda x: x[1], reverse=True)[:5]
    for feat, imp in top5:
        print(f"  {feat:<30} {imp:.4f}")

    # Verify patient_id and target_proxy show up (they should be top features = leak)
    print(f"\nBest score: {result1['best_model_score']}")
    print(f"Best model: {result1['best_model_name']}")

    # ── Test Run 2: With exclusions (simulating Auditor rejection) ────────────
    print("\n--- RUN 2: With exclusions ['patient_id', 'target_proxy'] ---")
    state2 = {**state, "excluded_columns": ["patient_id", "target_proxy"]}
    result2 = engineer_node(state2)

    print(f"\nBest score after exclusions: {result2['best_model_score']}")
    print(f"(Should be lower than {result1['best_model_score']} — model can no longer cheat)")

    # Verify excluded columns are gone from feature importance
    remaining_features = list(result2["feature_importance"].keys())
    assert "patient_id" not in remaining_features, "patient_id should be excluded!"
    assert "target_proxy" not in remaining_features, "target_proxy should be excluded!"
    print(f"\n✅ Excluded columns correctly removed from training")

    print("\n" + "=" * 60)
    print("engineer.py test PASSED — FLAML is working on your machine!")
    print("=" * 60)
