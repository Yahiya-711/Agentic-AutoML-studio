# =============================================================================
# state.py — Shared Memory for the Agentic AutoML Pipeline
# =============================================================================
#
# WHAT IS THIS FILE?
# ------------------
# In LangGraph, every agent (node) is a Python function that receives a
# dictionary and returns a dictionary. That dictionary is the STATE.
#
# Think of it like a whiteboard in a team meeting:
#   - The Orchestrator writes its plan on the board
#   - The Engineer reads the plan, executes it, writes results back
#   - The Auditor reads the results, writes its verdict back
#   - Everyone reads from and writes to the SAME whiteboard
#
# This file defines the SCHEMA of that whiteboard — what fields exist,
# what type each field is, and what it means.
#
# WHY TypedDict AND NOT A REGULAR DICT?
# -------------------------------------
# A regular dict has no type safety. You could typo "target_colum" instead
# of "target_column" and Python won't complain until the code crashes at
# runtime — possibly during your demo.
#
# TypedDict gives you:
#   1. IDE autocomplete — start typing state["tar..."] and it suggests the key
#   2. Static type checking — mypy / pylance catches typos before runtime
#   3. Self-documentation — anyone reading this file knows exactly what
#      the pipeline carries at every stage
#
# WHY NOT PYDANTIC?
# -----------------
# Pydantic BaseModel is great for API validation, but it adds overhead:
#   - Validation runs on every field assignment
#   - LangGraph's internal state merging works better with plain dicts
#   - TypedDict is zero-overhead at runtime (it's just type hints)
# For an agent state that gets read/written dozens of times per pipeline
# run, TypedDict is the right tool.
#
# =============================================================================

from typing import TypedDict, Optional, List, Dict, Tuple, Any

MAX_RETRIES = 2
class AgentState(TypedDict):
    """
    The single source of truth for the entire pipeline.
    Every agent reads from this and returns an updated version of it.

    Lifecycle of a pipeline run:
    ┌─────────────────────────────────────────────────────────┐
    │ STAGE 1: User Input (filled by dashboard.py)            │
    │   file_path, columns, dtypes, task_description          │
    │   excluded_columns = [], retry_count = 0                │
    │   everything else = None                                │
    ├─────────────────────────────────────────────────────────┤
    │ STAGE 2: After Orchestrator runs                        │
    │   + target_column, metric, time_limit, task_type        │
    ├─────────────────────────────────────────────────────────┤
    │ STAGE 3: After Engineer runs                            │
    │   + leaderboard, feature_importance,                    │
    │     best_model_score, best_model_name                   │
    ├─────────────────────────────────────────────────────────┤
    │ STAGE 4: After Auditor runs                             │
    │   + approved, critique                                  │
    │   excluded_columns may grow (leaked cols added)         │
    │   retry_count increments by 1                           │
    │                                                         │
    │   IF approved=False AND retry_count < MAX_RETRIES:      │
    │     → Loop back to STAGE 3 (Engineer retrain)           │
    │   ELSE:                                                 │
    │     → Pipeline ends, dashboard reads final state        │
    └─────────────────────────────────────────────────────────┘
    """

    # =========================================================================
    # SECTION 1 — INPUT (set once by the Streamlit dashboard, never changed)
    # =========================================================================

    file_path: str
    # Absolute path to the uploaded CSV on disk.
    # Example: "/tmp/customer_churn.csv"
    # WHY STORE THE PATH AND NOT THE DATAFRAME?
    # Storing a pandas DataFrame inside a dict that LangGraph serializes
    # can cause memory duplication. We store the path and let each agent
    # load the file fresh. On a 8GB machine, this matters.

    columns: List[str]
    # All column names extracted from the CSV header.
    # Example: ["age", "tenure", "monthly_charges", "churn"]
    # The Orchestrator uses this to decide which column to predict.
    # The Auditor uses this to validate that "leaked_columns" it
    # identifies actually exist in the dataset.

    dtypes: Dict[str, str]
    # Column name → pandas dtype as a string.
    # Example: {"age": "int64", "monthly_charges": "float64", "churn": "object"}
    # WHY STRINGIFY THE DTYPES?
    # pandas dtype objects are not JSON-serializable. LangGraph checkpoints
    # the state as JSON internally. String dtypes are safe to pass around.
    # The Orchestrator uses these to infer task type:
    #   "object" / low-cardinality int → classification
    #   continuous float/int → regression

    task_description: str
    # Free-text from the user describing what they want to predict.
    # Example: "predict whether a customer will churn next month"
    # This seeds the Orchestrator's LLM prompt with user intent.
    # If the user leaves it blank, dashboard.py fills it with a default.

    # =========================================================================
    # SECTION 2 — ORCHESTRATOR OUTPUT (filled after orchestrator_node runs)
    # =========================================================================

    target_column: Optional[str]
    # The column the model will predict. Chosen by the Orchestrator LLM.
    # Example: "churn"
    # Optional because it starts as None — the Orchestrator fills it.
    # The Auditor validates this is not in excluded_columns.

    metric: Optional[str]
    # The evaluation metric FLAML will optimize for.
    # Possible values: "roc_auc", "accuracy", "rmse", "mae", "r2"
    # The Orchestrator chooses based on task_type:
    #   binary classification  → "roc_auc"
    #   multiclass             → "accuracy"
    #   regression             → "rmse"

    time_limit: Optional[int]
    # Seconds FLAML is allowed to train. Set by Orchestrator, capped by UI.
    # Example: 60
    # On your i5-7th gen, 60 seconds trains 8-12 models comfortably.
    # Higher = more models tried = potentially better score, but slower demo.

    task_type: Optional[str]
    # "classification" or "regression" — inferred by Orchestrator.
    # FLAML needs this explicitly to know which algorithms to try.

    # =========================================================================
    # SECTION 3 — ENGINEER OUTPUT (filled after engineer_node runs)
    # =========================================================================

    leaderboard: Optional[List[Dict[str, Any]]]
    # Ranked list of all models FLAML trained, best first.
    # Each entry is a dict: {"model": "LGBMClassifier", "score": 0.9821,
    #                        "training_time": 12.3}
    # WHY List[Dict] AND NOT A DATAFRAME?
    # pandas DataFrames can't be stored in TypedDict cleanly and
    # won't serialize to JSON. List of dicts is native Python, JSON-safe,
    # and pd.DataFrame(state["leaderboard"]) rebuilds the table instantly.

    feature_importance: Optional[Dict[str, float]]
    # Column name → importance score from the best model.
    # Example: {"monthly_charges": 0.42, "tenure": 0.31, "age": 0.12,
    #           "customer_id": 0.09}
    # The Auditor inspects the TOP 5 entries here.
    # If "customer_id" shows up high, it's a leak signal.
    # FLAML gives this natively (no permutation recomputation needed).

    best_model_score: Optional[float]
    # Single float representing the best model's validation score.
    # Example: 0.9821 (roc_auc), 0.8534 (accuracy), 0.234 (rmse)
    # The Auditor mentions this in its prompt:
    # "Model scored 0.9821 on roc_auc — and the top feature is customer_id.
    #  Is this suspiciously high?"

    best_model_name: Optional[str]
    # Human-readable name of the winning model.
    # Example: "LGBMClassifier", "XGBClassifier", "RandomForestClassifier"
    # Displayed on the dashboard metrics row.

    # =========================================================================
    # SECTION 4 — AUDITOR OUTPUT (filled after auditor_node runs)
    # =========================================================================

    approved: Optional[bool]
    # The Auditor's final verdict.
    #   True  → model is clean, pipeline ends, show results
    #   False → leakage detected, add leaked cols to excluded_columns,
    #           loop back to Engineer IF retry_count < MAX_RETRIES
    # Starts as None. After first Auditor run it becomes True or False.

    critique: Optional[str]
    # One-sentence explanation from the Auditor LLM.
    # Example: "customer_id is a unique row identifier and causes data leakage."
    # Displayed in the Audit Report tab of the dashboard.
    # Even when approved=True, this may contain minor notes.

    # =========================================================================
    # SECTION 5 — LOOP CONTROL (updated by Auditor, read by graph router)
    # =========================================================================

    excluded_columns: List[str]
    # Columns to DROP before the next training run.
    # Starts as [] (empty list — no exclusions on first pass).
    # The Auditor appends to this list when it finds leakage.
    # The Engineer uses this list to filter the dataset.
    #
    # IMPORTANT: This list grows across retries — it is CUMULATIVE.
    # If loop 1 finds "customer_id" and loop 2 finds "target_flag",
    # after loop 2 excluded_columns = ["customer_id", "target_flag"].
    # We never remove entries — once excluded, always excluded.
    #
    # WHY NOT OPTIONAL?
    # The graph router checks `if not state["excluded_columns"]` to decide
    # whether to loop. If this field could be None, that check would pass
    # when it shouldn't. Initialising to [] in the dashboard guarantees
    # the router always has a list to check.

    retry_count: int
    # How many times the Engineer has been called so far.
    # Starts at 0. Incremented by 1 each time the Auditor runs.
    #
    # The graph router checks:
    #   if retry_count >= MAX_RETRIES (2): stop, even if not approved
    #
    # WHY THIS MATTERS ON YOUR MACHINE:
    # Without this cap, a buggy Auditor prompt that never returns
    # approved=True would loop forever, filling your RAM with repeated
    # FLAML training runs. The cap is a non-negotiable safety valve.
    #
    # WHY NOT OPTIONAL?
    # Same reason as excluded_columns — the router does integer comparison.
    # Initialise to 0 in the dashboard, never let it be None.


# =============================================================================
# HELPER: Initial state factory
# =============================================================================
# This function is called by dashboard.py to create the starting state.
# Centralizing it here means dashboard.py doesn't need to know which
# fields default to None vs [] vs 0 — it just calls this function.
# =============================================================================

def create_initial_state(
    file_path: str,
    columns: List[str],
    dtypes: Dict[str, str],
    task_description: str,
    time_limit: int = 60,
) -> AgentState:
    """
    Factory function that returns a fully-initialized AgentState dict.

    Usage in dashboard.py:
        state = create_initial_state(
            file_path="/tmp/churn.csv",
            columns=df.columns.tolist(),
            dtypes=df.dtypes.astype(str).to_dict(),
            task_description="predict customer churn",
            time_limit=60,
        )
        result = graph.invoke(state)

    WHY A FACTORY FUNCTION?
    If you add a new field to AgentState later (e.g., "model_explainability"),
    you only need to add the default in ONE place — here. Without this,
    you'd need to update every place that constructs a state dict.
    """
    return AgentState(
        # ── Input ─────────────────────────────
        file_path=file_path,
        columns=columns,
        dtypes=dtypes,
        task_description=task_description,

        # ── Orchestrator output (unfilled) ────
        target_column=None,
        metric=None,
        time_limit=time_limit,
        task_type=None,

        # ── Engineer output (unfilled) ────────
        leaderboard=None,
        feature_importance=None,
        best_model_score=None,
        best_model_name=None,

        # ── Auditor output (unfilled) ─────────
        approved=None,
        critique=None,

        # ── Loop control (initialised) ────────
        excluded_columns=[],   # empty — no exclusions on first pass
        retry_count=0,         # zero — no retries yet
    )
def validate_state(state: AgentState) -> Tuple[bool, str]:
    """
    Lightweight safety check before a node runs.
    Prevents catastrophic loops and corrupted state execution.
    """

    # Required base inputs
    if not state.get("file_path"):
        return False, "Missing file_path"

    if not state.get("columns"):
        return False, "Missing columns"

    # Retry safety
    retry_count = state.get("retry_count")

    if retry_count is None:
        return False, "retry_count is None"

    if retry_count >= MAX_RETRIES:
        return False, "Max retries reached"

    return True, "OK"


# =============================================================================
# QUICK SELF-TEST
# Run this file directly to verify the schema works:
#   python3 app/agents/state.py
# =============================================================================

if __name__ == "__main__":
    # Create a sample state and print it — confirms no import errors
    sample = create_initial_state(
        file_path="/tmp/test.csv",
        columns=["age", "tenure", "monthly_charges", "customer_id", "churn"],
        dtypes={
            "age": "int64",
            "tenure": "int64",
            "monthly_charges": "float64",
            "customer_id": "object",
            "churn": "object",
        },
        task_description="predict customer churn",
        time_limit=60,
    )

    print("=" * 60)
    print("AgentState initialized successfully!")
    print("=" * 60)
    print(f"  file_path:         {sample['file_path']}")
    print(f"  columns:           {sample['columns']}")
    print(f"  task_description:  {sample['task_description']}")
    print(f"  target_column:     {sample['target_column']}  ← None until Orchestrator runs")
    print(f"  excluded_columns:  {sample['excluded_columns']}  ← empty list, not None")
    print(f"  retry_count:       {sample['retry_count']}  ← 0, not None")
    print(f"  approved:          {sample['approved']}  ← None until Auditor runs")
    print("=" * 60)
    print("All fields present. state.py is working correctly.")
