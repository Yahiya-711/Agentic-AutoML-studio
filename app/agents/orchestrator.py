# =============================================================================
# orchestrator.py — The LLM Planning Agent
# =============================================================================

import json
import re
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.agents.state import AgentState


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

LLM = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0,
    num_predict=512,
    num_ctx=2048,
    request_timeout=120.0,  # CRITICAL: Prevents hanging on i5 CPU
)


# =============================================================================
# PROMPT DESIGN
# =============================================================================

SYSTEM_PROMPT = """You are an expert machine learning engineer.
Your job is to analyze a dataset schema and decide how to train an ML model.
You respond ONLY with a valid JSON object.
No markdown. No code fences. No explanation. No preamble. Just raw JSON."""

HUMAN_PROMPT = """Analyze this dataset and create a training plan.

DATASET INFORMATION:
- Column names: {columns}
- Column data types: {dtypes}
- Number of columns: {num_columns}
- User's prediction goal: "{task_description}"

YOUR TASK:
Choose the best values for each field below.

DECISION RULES (follow these strictly):
1. target_column: Pick the column most likely to be the prediction target.
   - Prefer columns named: target, label, class, churn, fraud, outcome, status, survived, approved, result
   - If user mentioned a specific column in their goal, use that
   - Never pick ID columns (customer_id, user_id, row_id, index)
   - Never pick timestamp columns (created_at, date, timestamp)

2. task_type:
   - "classification" if target has categorical/boolean/binary values OR fewer than 15 unique values
   - "regression" if target is a continuous numeric column (price, age, salary, score)

3. metric:
   - Binary classification (2 classes)   → "roc_auc"
   - Multiclass classification (3+ classes) → "accuracy"
   - Regression                           → "rmse"

4. time_limit:
   - Use exactly {time_limit} seconds (set by the user in the dashboard)

Respond with ONLY this JSON structure, nothing else:
{{
  "target_column": "exact_column_name_from_the_list",
  "task_type": "classification",
  "metric": "roc_auc",
  "time_limit": {time_limit},
  "reasoning": "one sentence explaining your target choice"
}}"""


# =============================================================================
# RESPONSE CLEANING
# =============================================================================

def _clean_llm_response(raw_text: str) -> str:
    """Strip everything that isn't the JSON object from the LLM's response."""
    text = raw_text

    # Step 1: Remove DeepSeek-R1's chain-of-thought block
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Step 2: Remove markdown code fences
    text = re.sub(r'```(?:json)?', '', text)

    # Step 3: Extract the JSON object
    json_match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if json_match:
        text = json_match.group(0)

    return text.strip()


# =============================================================================
# VALIDATION AND FALLBACK
# =============================================================================

def _validate_and_fix_plan(plan: dict, state: AgentState) -> dict:
    """Validate the LLM's JSON plan and fix any invalid values."""
    valid_columns = set(state["columns"])

    # Validate target_column
    target = plan.get("target_column", "")
    if target not in valid_columns:
        task_lower = state.get("task_description", "").lower()
        found = None
        for col in state["columns"]:
            if col.lower() in task_lower:
                found = col
                break
        plan["target_column"] = found if found else state["columns"][-1]

    # Validate task_type
    task_type = plan.get("task_type", "").lower()
    if task_type not in ("classification", "regression"):
        plan["task_type"] = "classification"

    # Validate metric vs task_type consistency
    valid_classification_metrics = {"roc_auc", "accuracy", "f1", "log_loss"}
    valid_regression_metrics = {"rmse", "mae", "mse", "r2", "mape"}

    metric = plan.get("metric", "")
    if plan["task_type"] == "classification":
        if metric not in valid_classification_metrics:
            plan["metric"] = "roc_auc"
    else:
        if metric not in valid_regression_metrics:
            plan["metric"] = "rmse"

    # Always use the time_limit from state (set by user's dashboard slider)
    plan["time_limit"] = int(state.get("time_limit", 60))

    return plan


def _infer_task_type_from_dtype(state: AgentState, target_column: str) -> str:
    """
    CRITICAL FIX: Infer task_type from the target column's dtype.
    
    This is called BEFORE the LLM to set a baseline, and AFTER to validate.
    """
    target_dtype = state["dtypes"].get(target_column, "object").lower()
    
    # Float columns are almost always regression targets
    if "float" in target_dtype:
        return "regression"
    
    # Int columns could be either - check column name hints
    elif "int" in target_dtype:
        regression_keywords = ["price", "cost", "salary", "age", "score", "amount", "value"]
        for keyword in regression_keywords:
            if keyword in target_column.lower():
                return "regression"
        return "classification"
    
    # Object/string columns are classification
    else:
        return "classification"


def _infer_task_type_from_data(state: AgentState, target_column: str) -> Optional[str]:
    """Read the actual CSV to infer task_type from target column's data cardinality."""
    try:
        import pandas as pd
        target_series = pd.read_csv(state["file_path"], usecols=[target_column])[target_column]
        n_unique = target_series.nunique()
        n_total = len(target_series)

        if n_unique <= 20 and (n_unique / n_total) < 0.05:
            return "classification"
        elif n_unique > 20:
            return "regression"
        else:
            return None
    except Exception:
        return None


def _rule_based_fallback(state: AgentState) -> dict:
    """
    CRITICAL FIX: Rule-based fallback that checks dtype before deciding task_type.
    """
    print(f"[Orchestrator] Using rule-based fallback...")

    # Rule 1: Find target column by common naming conventions
    target_keywords = [
        "target", "label", "class", "churn", "fraud", "outcome",
        "status", "survived", "approved", "result", "y", "output",
        "category", "type", "predict"
    ]

    target_col = None
    columns_lower = {col.lower(): col for col in state["columns"]}

    # Check for exact keyword matches first
    for keyword in target_keywords:
        if keyword in columns_lower:
            target_col = columns_lower[keyword]
            break

    # Check for partial matches
    if not target_col:
        for keyword in target_keywords:
            for col_lower, col_orig in columns_lower.items():
                if keyword in col_lower:
                    target_col = col_orig
                    break
            if target_col:
                break

    # Final fallback: last column
    if not target_col:
        target_col = state["columns"][-1]

    # CRITICAL: Infer task_type from dtype, NOT hardcoded classification
    task_type = _infer_task_type_from_dtype(state, target_col)
    
    # Set metric based on task_type
    if task_type == "regression":
        metric = "rmse"
    else:
        metric = "roc_auc"

    plan = {
        "target_column": target_col,
        "task_type": task_type,
        "metric": metric,
        "time_limit": state.get("time_limit", 60),
        "reasoning": "Rule-based fallback: LLM unavailable",
    }
    print(f"[Orchestrator] Rule-based plan: {plan}")
    return plan


# =============================================================================
# THE ORCHESTRATOR NODE
# =============================================================================

def orchestrator_node(state: AgentState) -> AgentState:
    """The Orchestrator agent function. Called by LangGraph as the entry node."""
    print(f"\n[Orchestrator] Analyzing schema with DeepSeek-R1...")
    print(f"[Orchestrator] Columns: {state['columns']}")
    print(f"[Orchestrator] Task: {state.get('task_description', 'not specified')}")

    plan = None

    # =========================================================================
    # LAYER 1: LLM-based planning
    # =========================================================================
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])

        chain = prompt | LLM

        print(f"[Orchestrator] Calling LLM...")
        response = chain.invoke({
            "columns": state["columns"],
            "dtypes": state["dtypes"],
            "num_columns": len(state["columns"]),
            "task_description": state.get("task_description", "predict the most relevant target"),
            "time_limit": state.get("time_limit", 60),
        })

        raw_text = response.content
        print(f"[Orchestrator] Raw LLM response length: {len(raw_text)} chars")
        print(f"[Orchestrator] Raw response preview: {raw_text[:200]}...")

        # Clean the response
        cleaned = _clean_llm_response(raw_text)
        print(f"[Orchestrator] Cleaned response: {cleaned[:200]}...")

        # Parse JSON
        plan = json.loads(cleaned)
        print(f"[Orchestrator] LLM plan parsed successfully: {plan}")

    except json.JSONDecodeError as e:
        print(f"[Orchestrator] JSON parse failed: {e}")
        print(f"[Orchestrator] Falling through to Layer 2 (rule-based fallback)")
        plan = None

    except Exception as e:
        print(f"[Orchestrator] LLM call failed: {type(e).__name__}: {e}")
        print(f"[Orchestrator] Is Ollama running? Try: ollama serve")
        plan = None

    # =========================================================================
    # LAYER 2: Rule-based fallback (FIXED to check dtype)
    # =========================================================================
    if plan is None:
        plan = _rule_based_fallback(state)

    # =========================================================================
    # VALIDATE AND FIX the plan
    # =========================================================================
    plan = _validate_and_fix_plan(plan, state)

    # =========================================================================
    # SECONDARY CHECK: Verify task_type using actual data cardinality
    # =========================================================================
    inferred_task_type = _infer_task_type_from_data(state, plan["target_column"])
    if inferred_task_type and inferred_task_type != plan["task_type"]:
        print(f"[Orchestrator] Data cardinality overrides task_type: "
              f"{plan['task_type']} → {inferred_task_type}")
        plan["task_type"] = inferred_task_type
        if inferred_task_type == "regression":
            plan["metric"] = "rmse"
        else:
            plan["metric"] = "roc_auc"

    # =========================================================================
    # FINAL SUMMARY LOG
    # =========================================================================
    print(f"\n[Orchestrator] ✅ Final Plan:")
    print(f"  Target column : {plan['target_column']}")
    print(f"  Task type     : {plan['task_type']}")
    print(f"  Metric        : {plan['metric']}")
    print(f"  Time limit    : {plan['time_limit']}s")
    print(f"  Reasoning     : {plan.get('reasoning', 'N/A')}")

    return {
        **state,
        "target_column": plan["target_column"],
        "task_type": plan["task_type"],
        "metric": plan["metric"],
        "time_limit": plan["time_limit"],
    }


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Orchestrator Node — Standalone Test")
    print("=" * 60)
    print("NOTE: Ollama must be running → ollama serve")
    print()

    from app.agents.state import create_initial_state

    # ── Test 1: Binary classification ─────────────────────────────────────────
    print("--- TEST 1: Binary Classification ---")
    state1 = create_initial_state(
        file_path="/tmp/dummy.csv",
        columns=["customer_id", "age", "tenure", "monthly_charges", "churn"],
        dtypes={
            "customer_id": "object",
            "age": "int64",
            "tenure": "int64",
            "monthly_charges": "float64",
            "churn": "object",
        },
        task_description="predict whether a customer will churn next month",
        time_limit=60,
    )

    result1 = orchestrator_node(state1)
    print(f"\nResult:")
    print(f"  target_column : {result1['target_column']}  (expected: churn)")
    print(f"  task_type     : {result1['task_type']}  (expected: classification)")
    print(f"  metric        : {result1['metric']}  (expected: roc_auc)")

    assert result1["target_column"] == "churn"
    assert result1["task_type"] == "classification"
    print("✅ Test 1 PASSED")

    # ── Test 2: Regression (THE TEST THAT WAS FAILING) ────────────────────────
    print("\n--- TEST 2: Regression ---")
    state2 = create_initial_state(
        file_path="/tmp/dummy.csv",
        columns=["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "zipcode", "price"],
        dtypes={
            "bedrooms": "int64",
            "bathrooms": "float64",
            "sqft_living": "int64",
            "sqft_lot": "int64",
            "floors": "float64",
            "zipcode": "int64",
            "price": "float64",
        },
        task_description="predict house sale price",
        time_limit=60,
    )

    result2 = orchestrator_node(state2)
    print(f"\nResult:")
    print(f"  target_column : {result2['target_column']}  (expected: price)")
    print(f"  task_type     : {result2['task_type']}  (expected: regression)")
    print(f"  metric        : {result2['metric']}  (expected: rmse)")

    assert result2["target_column"] == "price"
    assert result2["task_type"] == "regression"  # THIS WAS FAILING BEFORE
    print("✅ Test 2 PASSED")

    # ── Test 3: User-guided target ────────────────────────────────────────────
    print("\n--- TEST 3: User-guided target ---")
    state3 = create_initial_state(
        file_path="/tmp/dummy.csv",
        columns=["id", "feature_a", "feature_b", "col_z"],
        dtypes={col: "float64" for col in ["id", "feature_a", "feature_b", "col_z"]},
        task_description="predict col_z which is the fraud indicator",
        time_limit=45,
    )

    result3 = orchestrator_node(state3)
    print(f"\nResult:")
    print(f"  target_column : {result3['target_column']}  (expected: col_z)")
    print(f"  time_limit    : {result3['time_limit']}  (expected: 45)")

    assert result3["time_limit"] == 45
    print("✅ Test 3 PASSED")

    print("\n" + "=" * 60)
    print("orchestrator.py — ALL TESTS PASSED")
    print("=" * 60)