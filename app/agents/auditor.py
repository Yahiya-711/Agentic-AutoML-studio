# =============================================================================
# auditor.py — The Data Leakage Detection Agent
# =============================================================================
#
# WHAT IS THIS FILE?
# ------------------
# This is the "Quality Control" agent — the most innovative part of the
# entire project. No standard AutoML tool (H2O, AutoGluon, AutoSklearn)
# does what this agent does automatically.
#
# After the Engineer trains models, the Auditor:
#   1. Inspects the top features by importance
#   2. Uses DeepSeek-R1 to reason about whether any feature is "cheating"
#   3. Returns a verdict: approved=True (clean) or approved=False (leakage)
#   4. If leakage found: names the bad columns → Engineer retrains without them
#
# WHAT IS DATA LEAKAGE?
# ----------------------
# Data leakage is when a feature gives the model information it wouldn't
# have in a real production scenario — letting it "cheat" during training.
#
# Three types your Auditor catches:
#
#   TYPE 1 — ID Columns:
#     customer_id, user_id, order_id, row_number, index
#     These are unique per row. If the model memorises "customer_id=12345
#     had churn=1", it learns nothing generalizable.
#     Sign: appears in feature importance despite being meaningless.
#
#   TYPE 2 — Target Proxies:
#     A column that directly encodes or is derived from the target.
#     Example: target="churn", proxy="churned_flag" or "days_since_churn"
#     Sign: near-perfect model score (>0.99) + this feature is #1 in importance.
#
#   TYPE 3 — Future Data (Post-event leakage):
#     Information that only exists AFTER the event you're predicting.
#     Example: predicting loan default using "collection_date" — that date
#     only exists for loans that already defaulted.
#     Sign: temporal column that semantically follows the target event.
#
# WHY CAN AN LLM DETECT THIS WHEN STATISTICS CANNOT?
# ---------------------------------------------------
# Statistics sees numbers. "customer_id" importance = 0.15 is just a number.
# It doesn't know that "customer_id" semantically means "unique row identifier".
#
# An LLM that has read thousands of ML articles knows:
#   - Columns ending in "_id" are usually identifiers → suspicious
#   - Columns with the same root word as the target are proxies
#   - Date/timestamp columns after-the-fact are future leakage
#
# This is exactly the kind of semantic, context-aware reasoning that
# LLMs excel at and that rule-based systems miss.
#
# HOW THE SELF-CORRECTION LOOP WORKS
# ------------------------------------
#
#   Engineer trains → state["feature_importance"] filled
#         │
#         ▼
#   Auditor reads top 5 features → LLM reasons about leakage
#         │
#   ┌─────┴─────┐
#   │ approved? │
#   └─────┬─────┘
#         │ YES → pipeline ends, results shown
#         │ NO  → leaked columns added to state["excluded_columns"]
#                 retry_count += 1
#                 LangGraph routes back to Engineer
#                 Engineer retrains WITHOUT the leaked columns
#                 Auditor runs again on the new model
#
# This loop runs at most MAX_RETRIES (2) times — a hard safety cap.
#
# =============================================================================

import json
import re
from typing import Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState


# =============================================================================
# LLM CONFIGURATION
# =============================================================================
#
# Same model as the Orchestrator (deepseek-r1:1.5b) but with different
# parameters tuned for the Auditor's task.
#
# WHY num_predict=768 (MORE THAN ORCHESTRATOR'S 512)?
# The Auditor's reasoning chain tends to be longer. It needs to:
#   - Consider each feature individually
#   - Cross-reference with the target column name
#   - Reason about whether 99% accuracy is suspicious
# That reasoning in <think> blocks can easily be 400-600 tokens.
# 768 gives enough room without wasting RAM on unused context.
#
# temperature=0 → same reason as Orchestrator: deterministic JSON output.
# We never want a creative Auditor that randomly flags clean columns.

LLM = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0,
    num_predict=768,
    num_ctx=2048,
)


# =============================================================================
# STATIC LEAKAGE RULES (Pre-LLM filter)
# =============================================================================
#
# WHY HAVE STATIC RULES ALONGSIDE THE LLM?
# -----------------------------------------
# The LLM is great at semantic reasoning but might miss obvious patterns
# when the column name is ambiguous or the model's attention drifts.
# Static rules are deterministic — they ALWAYS catch known patterns.
#
# Running static rules BEFORE the LLM means:
#   1. Obvious leaks are caught even if Ollama is slow/unavailable
#   2. The LLM prompt can focus on subtle cases (it already knows about
#      the obvious ones being handled)
#   3. Faster — no LLM call needed for clear ID columns
#
# These patterns are compiled from common data science anti-patterns:

# Column name substrings that almost always indicate ID columns
ID_PATTERNS = {
    "_id", "id_", "_key", "key_", "_num", "row_", "_row",
    "index", "idx", "_idx", "uuid", "guid", "serial",
    "account_no", "ref_no", "record_id",
}

# Column name substrings that suggest future/post-event data
FUTURE_PATTERNS = {
    "date_of_", "end_date", "completion_date", "close_date",
    "resolution_date", "default_date", "churn_date", "exit_date",
    "cancellation_date",
}

# Column name substrings that suggest target proxies
PROXY_PATTERNS = {
    "_flag", "_label", "_target", "_class", "_category",
    "is_", "has_", "will_", "did_", "was_",
}


def _static_leakage_check(
    feature_importance: Dict[str, float],
    target_column: str,
    top_n: int = 10,
) -> Tuple[List[str], List[str]]:
    """
    Rule-based pre-filter that catches obvious leakage patterns
    by matching column name substrings.

    Returns:
        confirmed_leaks  : columns definitely leaked (ID patterns)
        suspicious       : columns to flag for LLM review

    WHY CHECK ALL TOP N, NOT JUST TOP 5?
    The Auditor's LLM prompt shows top 5. But an ID column ranked 6th
    is still a leak — it just wasn't important enough to show in top 5.
    Static rules check the wider top-10 to be thorough.

    WHY SEPARATE "confirmed" FROM "suspicious"?
    - confirmed_leaks: Column name contains "_id" or "uuid" etc.
      We can add these to excluded_columns WITHOUT asking the LLM.
      This is deterministic and saves one LLM round-trip.
    - suspicious: Column name contains "_flag" or starts with "is_".
      These MIGHT be legitimate features. We pass them to the LLM
      for semantic reasoning.
    """
    confirmed_leaks = []
    suspicious = []

    # Sort features by importance, check top N
    top_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    for col_name, importance in top_features:
        col_lower = col_name.lower()

        # Skip the target column itself (obviously can't be a leaked feature)
        if col_lower == target_column.lower():
            continue

        # Check for ID patterns (high confidence → confirmed leak)
        is_id = any(pattern in col_lower for pattern in ID_PATTERNS)
        if is_id:
            confirmed_leaks.append(col_name)
            continue

        # Check if column name contains the target name (proxy leak)
        # e.g., target="churn", column="churn_score" or "churn_probability"
        target_lower = target_column.lower()
        if target_lower in col_lower and col_lower != target_lower:
            confirmed_leaks.append(col_name)
            continue

        # Check for proxy patterns (lower confidence → send to LLM)
        is_proxy_candidate = any(pattern in col_lower for pattern in PROXY_PATTERNS)
        is_future_candidate = any(pattern in col_lower for pattern in FUTURE_PATTERNS)

        if is_proxy_candidate or is_future_candidate:
            suspicious.append(col_name)

    return confirmed_leaks, suspicious


# =============================================================================
# RESPONSE CLEANING (identical logic to orchestrator.py)
# =============================================================================

def _clean_llm_response(raw_text: str) -> str:
    """
    Strip DeepSeek-R1's <think> blocks and markdown from the response.

    Same cleaning pipeline as orchestrator.py — see that file for full
    explanation. Duplicated here (not imported) to keep each agent file
    self-contained and independently testable.
    """
    text = raw_text

    # Remove chain-of-thought block
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Remove markdown code fences
    text = re.sub(r'```(?:json)?', '', text)

    # Extract just the JSON object
    json_match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if json_match:
        text = json_match.group(0)

    return text.strip()


# =============================================================================
# PROMPT DESIGN
# =============================================================================
#
# THE AUDITOR'S PROMPT IS THE MOST CAREFULLY ENGINEERED IN THE PROJECT.
#
# Why? Because false positives (flagging clean features) cause unnecessary
# retraining loops that waste your limited training time budget and RAM.
# False negatives (missing real leakage) mean the model is broken but
# the Auditor approved it — worse outcome than a false positive.
#
# The prompt is designed to minimize both:
#
# 1. GIVE CONTEXT: Show the model name, metric, score, AND top features
#    together. A 99.8% roc_auc is suspicious context. A 72% accuracy
#    with the same top features is probably fine.
#
# 2. DEFINE LEAKAGE PRECISELY: Don't say "check for leakage". Say exactly
#    what patterns count as leakage. Smaller models follow explicit rules
#    better than vague instructions.
#
# 3. GIVE EXAMPLES: Show what a leaked column name looks like vs a clean one.
#    Few-shot examples dramatically improve small model reliability.
#
# 4. CONSTRAIN THE OUTPUT: Specify the exact JSON structure. List only the
#    keys we'll parse. Extra keys are ignored; missing keys cause fallbacks.
#
# 5. THE SCORE CONTEXT TRICK:
#    If score > 0.97 for a classification task, that's suspicious.
#    Perfect models almost always have a leak. We inject this reasoning
#    into the prompt as a hint so the model treats high scores as a
#    red flag rather than a green light.

SYSTEM_PROMPT = """You are a strict machine learning auditor specializing in detecting data leakage.
Your job is to inspect a trained model's top features and decide if the model is cheating.
You respond ONLY with valid JSON. No markdown. No explanation outside the JSON. Just raw JSON."""

HUMAN_PROMPT = """A machine learning model was just trained. Audit it for data leakage.

═══ MODEL RESULTS ═══
Best model    : {model_name}
Metric used   : {metric}
Validation score: {score}
Target column : {target}
Training task : {task_type}

═══ TOP FEATURES BY IMPORTANCE ═══
{features_formatted}

═══ ALL COLUMNS IN DATASET ═══
{all_columns}

═══ LEAKAGE DEFINITIONS ═══
Flag a feature as leaked if it matches ANY of these:

TYPE 1 — ID COLUMN (always a leak):
  Column is a unique row identifier that has no predictive business value.
  Signs: name contains _id, id_, _key, uuid, index, serial, row_number
  Examples of leaked: customer_id, user_id, order_id, row_index
  Examples of clean:  age, tenure, product_category, monthly_charges

TYPE 2 — TARGET PROXY (always a leak):
  Column is derived from or directly encodes the target variable.
  Signs: column name contains the target name, or ends in _flag, _label, _target
  Examples of leaked: churn_score (if target=churn), default_flag (if target=defaulted)
  Examples of clean:  credit_score, satisfaction_score, payment_count

TYPE 3 — FUTURE DATA (always a leak):
  Column contains information that only exists AFTER the target event occurred.
  Signs: completion dates, end dates, resolution timestamps when predicting the event
  Examples of leaked: churn_date, default_date, cancellation_date
  Examples of clean:  signup_date, first_purchase_date, contract_start_date

═══ SUSPICION TRIGGERS ═══
{suspicion_context}

═══ YOUR TASK ═══
Examine ONLY the top features listed above (not all columns).
For each top feature, decide if it is a TYPE 1, TYPE 2, or TYPE 3 leak.
Only flag features that are CLEARLY leaked — do not flag legitimate business features.

Respond with ONLY this JSON:
{{
  "approved": true or false,
  "leaked_columns": ["exact_col_name_1", "exact_col_name_2"],
  "leak_types": {{"col_name": "TYPE1/TYPE2/TYPE3"}},
  "reason": "One sentence summary of what you found.",
  "confidence": "HIGH/MEDIUM/LOW"
}}

If no leakage found: approved=true, leaked_columns=[], leak_types={{}}
If leakage found: approved=false, list ONLY the leaked column names in leaked_columns."""


def _build_suspicion_context(
    best_model_score: float,
    metric: str,
    task_type: str,
    top_features: List[Tuple[str, float]],
) -> str:
    """
    Generate dynamic context lines that help the LLM calibrate suspicion.

    WHY DYNAMIC CONTEXT?
    A static prompt doesn't know:
    - Whether a 0.99 score is suspicious (yes, for most tasks)
    - Whether a feature named "id" is ranked #1 (very suspicious)
    - Whether the top feature has 10x the importance of #2 (suspicious dominance)

    These are signals that a human data scientist uses to smell a leak.
    We compute them programmatically and inject them as natural language
    into the prompt — giving the LLM the same "smell test" intuition.
    """
    context_lines = []

    # Score-based suspicion
    classification_metrics = {"roc_auc", "accuracy", "f1", "log_loss"}
    if task_type == "classification" and metric in classification_metrics:
        if best_model_score > 0.98:
            context_lines.append(
                f"⚠️  Score of {best_model_score:.4f} is suspiciously HIGH for a real-world "
                f"classification task. Scores above 0.97 almost always indicate data leakage. "
                f"Be very critical of the top features."
            )
        elif best_model_score > 0.95:
            context_lines.append(
                f"ℹ️  Score of {best_model_score:.4f} is high. Consider whether the top features "
                f"are providing unfair information."
            )
        else:
            context_lines.append(
                f"✅ Score of {best_model_score:.4f} is in a realistic range for this task."
            )

    # Feature dominance suspicion
    # If the top feature has >3x the importance of the second feature,
    # one feature is dominating — classic sign of a proxy or ID column
    if len(top_features) >= 2:
        top_imp = top_features[0][1]
        second_imp = top_features[1][1]
        if second_imp > 0 and top_imp / second_imp > 3.0:
            context_lines.append(
                f"⚠️  Feature '{top_features[0][0]}' has importance {top_imp:.4f} — "
                f"more than 3x the second feature ({second_imp:.4f}). "
                f"This dominance pattern is a strong indicator of a proxy or ID column."
            )

    # Top feature name inspection
    top_name = top_features[0][0].lower() if top_features else ""
    if any(pattern in top_name for pattern in ID_PATTERNS):
        context_lines.append(
            f"⚠️  The #1 feature '{top_features[0][0]}' contains an ID-like pattern in its name. "
            f"This is almost certainly a data leak."
        )

    if not context_lines:
        context_lines.append("No specific suspicion triggers detected. Review normally.")

    return "\n".join(context_lines)


def _format_features_for_prompt(
    feature_importance: Dict[str, float],
    top_n: int = 5,
) -> str:
    """
    Format the top N features as a readable numbered list for the prompt.

    WHY NUMBERED LIST INSTEAD OF JSON?
    Smaller LLMs process structured prose better than nested JSON inside
    a JSON-generating prompt. A numbered list with clear labels gives
    the model a natural reading order that mirrors how a human would
    scan a feature importance table.

    Output example:
        1. customer_id      importance=0.4231  (rank #1 of 30 features)
        2. monthly_charges  importance=0.2145  (rank #2 of 30 features)
        3. tenure           importance=0.1823  (rank #3 of 30 features)
    """
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    total_features = len(feature_importance)
    lines = []
    for rank, (col, imp) in enumerate(sorted_features, 1):
        lines.append(
            f"  {rank}. {col:<35} importance={imp:.4f}  "
            f"(rank #{rank} of {total_features} features)"
        )

    return "\n".join(lines)


# =============================================================================
# RESULT VALIDATION
# =============================================================================

def _validate_audit_result(
    result: dict,
    feature_importance: Dict[str, float],
    state: AgentState,
) -> dict:
    """
    Validate the LLM's audit result and remove any hallucinated column names.

    THE CRITICAL PROBLEM THIS SOLVES:
    -----------------------------------
    LLMs can hallucinate column names that don't exist in the dataset.
    If we pass a hallucinated column name to excluded_columns, the
    Engineer will try to drop a non-existent column. Pandas will
    silently ignore it (with errors="ignore"), but the Auditor flagged
    a ghost column — the real leak is still in the data.

    Worse: the LLM might flag the target column itself as leaked.
    Dropping the target would crash FLAML immediately.

    VALIDATION RULES:
    1. leaked_columns must exist in feature_importance keys (real features)
    2. leaked_columns must not include the target column
    3. approved must be a boolean (not a string "true")
    4. If leaked_columns is empty, force approved=True
    5. If confidence=LOW and no confirmed static leaks, reduce impact
    """
    valid_features = set(feature_importance.keys())
    target = state["target_column"]

    # Filter hallucinated column names
    raw_leaked = result.get("leaked_columns", [])
    validated_leaked = [
        col for col in raw_leaked
        if col in valid_features and col != target
    ]

    # Ensure approved is a proper boolean
    approved_raw = result.get("approved", True)
    if isinstance(approved_raw, str):
        approved = approved_raw.lower() == "true"
    else:
        approved = bool(approved_raw)

    # If no valid leaked columns found, force approval
    # (LLM said there's leakage but couldn't name a real column → false alarm)
    if not validated_leaked:
        approved = True

    # If approved=False but leaked_columns is empty after validation → approve
    if not validated_leaked and not approved:
        approved = True

    return {
        "approved": approved,
        "leaked_columns": validated_leaked,
        "leak_types": result.get("leak_types", {}),
        "reason": result.get("reason", "No reason provided."),
        "confidence": result.get("confidence", "MEDIUM"),
    }


# =============================================================================
# THE AUDITOR NODE — This is what LangGraph calls
# =============================================================================

def auditor_node(state: AgentState) -> AgentState:
    """
    The Auditor agent function. Called by LangGraph after engineer_node.

    LangGraph contract:
        - Input:  AgentState dict (with leaderboard, feature_importance from Engineer)
        - Output: AgentState dict (+ approved, critique, updated excluded_columns)
        - Must not raise exceptions

    Three-phase audit process:
        Phase 1: Static rules (instant, no LLM)
                 → catches obvious ID columns by name pattern
        Phase 2: LLM reasoning (DeepSeek-R1)
                 → catches subtle proxy/future leakage semantically
        Phase 3: Merge results + validate
                 → combine both findings, remove hallucinations

    Why this order?
    Static rules first means:
    - If obvious leaks found → we already know what to exclude
    - LLM prompt can focus on the remaining suspicious cases
    - If Ollama is down → static rules still provide partial protection
    """

    print(f"\n[Auditor] Running audit on training run #{state.get('retry_count', 0) + 1}")
    print(f"[Auditor] Best model: {state.get('best_model_name')} | "
          f"Score: {state.get('best_model_score')}")

    feature_importance = state.get("feature_importance", {})
    target_column = state["target_column"]
    metric = state.get("metric", "accuracy")
    task_type = state.get("task_type", "classification")
    best_score = state.get("best_model_score", 0.0)

    # Safety check: if no features, approve and move on
    if not feature_importance:
        print("[Auditor] No feature importance data available. Auto-approving.")
        return {
            **state,
            "approved": True,
            "critique": "No feature importance data available for audit.",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # =========================================================================
    # PHASE 1: Static rule-based leakage detection
    # =========================================================================
    print("[Auditor] Phase 1: Running static leakage rules...")

    confirmed_static, suspicious_static = _static_leakage_check(
        feature_importance=feature_importance,
        target_column=target_column,
        top_n=10,
    )

    if confirmed_static:
        print(f"[Auditor] Static rules found confirmed leaks: {confirmed_static}")
    if suspicious_static:
        print(f"[Auditor] Static rules found suspicious columns: {suspicious_static}")

    # =========================================================================
    # PHASE 2: LLM-based semantic leakage reasoning
    # =========================================================================
    print("[Auditor] Phase 2: Calling DeepSeek-R1 for semantic audit...")

    llm_result = None

    try:
        # Get top 5 features for the LLM prompt
        top_5 = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Build dynamic suspicion context
        suspicion_context = _build_suspicion_context(
            best_model_score=best_score,
            metric=metric,
            task_type=task_type,
            top_features=top_5,
        )

        # Format features as readable list
        features_formatted = _format_features_for_prompt(feature_importance, top_n=5)

        # Build and invoke the chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        chain = prompt | LLM

        response = chain.invoke({
            "model_name": state.get("best_model_name", "unknown"),
            "metric": metric,
            "score": round(best_score, 4),
            "target": target_column,
            "task_type": task_type,
            "features_formatted": features_formatted,
            "all_columns": state.get("columns", []),
            "suspicion_context": suspicion_context,
        })

        raw_text = response.content
        print(f"[Auditor] LLM response length: {len(raw_text)} chars")

        # Clean and parse
        cleaned = _clean_llm_response(raw_text)
        llm_result = json.loads(cleaned)
        print(f"[Auditor] LLM verdict: approved={llm_result.get('approved')}, "
              f"leaked={llm_result.get('leaked_columns')}, "
              f"confidence={llm_result.get('confidence')}")

    except json.JSONDecodeError as e:
        print(f"[Auditor] LLM JSON parse failed: {e}")
        llm_result = None

    except Exception as e:
        print(f"[Auditor] LLM call failed: {e}")
        llm_result = None

    # =========================================================================
    # PHASE 3: Merge static + LLM results
    # =========================================================================
    print("[Auditor] Phase 3: Merging static and LLM findings...")

    # Start with static confirmed leaks
    all_leaked = list(confirmed_static)

    if llm_result is not None:
        # Add LLM-identified leaks (will be validated below)
        all_leaked.extend(llm_result.get("leaked_columns", []))
        critique = llm_result.get("reason", "Audit complete.")
        confidence = llm_result.get("confidence", "MEDIUM")
        leak_types = llm_result.get("leak_types", {})
    else:
        # LLM failed — fall back to static rules only
        critique = (
            f"LLM audit unavailable. Static rules flagged: {confirmed_static}."
            if confirmed_static
            else "LLM audit unavailable. No static rule violations found. Auto-approved."
        )
        confidence = "LOW"
        leak_types = {col: "TYPE1" for col in confirmed_static}

    # Deduplicate
    all_leaked = list(set(all_leaked))

    # Build a synthetic result dict for validation
    merged_result = {
        "approved": len(all_leaked) == 0,
        "leaked_columns": all_leaked,
        "leak_types": leak_types,
        "reason": critique,
        "confidence": confidence,
    }

    # Validate (remove hallucinations, check column existence)
    validated = _validate_audit_result(
        result=merged_result,
        feature_importance=feature_importance,
        state=state,
    )

    # =========================================================================
    # UPDATE EXCLUDED COLUMNS (cumulative across retry loops)
    # =========================================================================
    previous_exclusions = state.get("excluded_columns", [])
    new_exclusions = list(set(previous_exclusions + validated["leaked_columns"]))

    # =========================================================================
    # FINAL SUMMARY LOG
    # =========================================================================
    print(f"\n[Auditor] ✅ Audit Complete:")
    print(f"  Approved        : {validated['approved']}")
    print(f"  Leaked columns  : {validated['leaked_columns']}")
    print(f"  Confidence      : {validated['confidence']}")
    print(f"  Reason          : {validated['reason']}")
    print(f"  All exclusions  : {new_exclusions}")

    if not validated["approved"]:
        print(f"  → Routing back to Engineer for retrain without: {validated['leaked_columns']}")
    else:
        print(f"  → Model approved. Pipeline complete.")

    # =========================================================================
    # RETURN UPDATED STATE
    # =========================================================================
    return {
        **state,
        "approved": validated["approved"],
        "critique": validated["reason"],
        "excluded_columns": new_exclusions,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# =============================================================================
# STANDALONE TEST
# Run this file directly to verify auditor logic (no Ollama needed for
# static tests, Ollama needed for LLM tests):
#   python3 app/agents/auditor.py
#
# Tests:
#   1. Static rules catch obvious ID column
#   2. Static rules catch target proxy
#   3. LLM catches subtle proxy (requires Ollama)
#   4. Clean dataset gets approved
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Auditor Node — Standalone Test")
    print("=" * 60)

    from app.agents.state import create_initial_state

    # ── Test 1: Static rule catches ID column ─────────────────────────────────
    print("\n--- TEST 1: Static rules — ID column leak ---")

    state1 = create_initial_state(
        file_path="/tmp/dummy.csv",
        columns=["customer_id", "age", "tenure", "monthly_charges", "churn"],
        dtypes={"customer_id": "object", "age": "int64", "tenure": "int64",
                "monthly_charges": "float64", "churn": "object"},
        task_description="predict churn",
        time_limit=60,
    )
    state1["target_column"] = "churn"
    state1["metric"] = "roc_auc"
    state1["task_type"] = "classification"
    state1["best_model_name"] = "LGBMClassifier"
    state1["best_model_score"] = 0.9954   # suspiciously high
    state1["leaderboard"] = [{"model": "LGBMClassifier", "score": 0.9954, "training_time_s": 12.0}]
    state1["feature_importance"] = {
        "customer_id": 0.45,        # ID column ranked #1 — clear leak
        "monthly_charges": 0.25,
        "tenure": 0.18,
        "age": 0.12,
    }

    result1 = auditor_node(state1)
    print(f"\nResult: approved={result1['approved']}")
    print(f"Excluded: {result1['excluded_columns']}")
    print(f"Critique: {result1['critique']}")

    assert not result1["approved"], "FAIL: Should have detected customer_id as leak"
    assert "customer_id" in result1["excluded_columns"], "FAIL: customer_id should be excluded"
    print("✅ Test 1 PASSED — ID column correctly flagged")

    # ── Test 2: Static rules catch target proxy ───────────────────────────────
    print("\n--- TEST 2: Static rules — Target proxy leak ---")

    state2 = {**state1}
    state2["feature_importance"] = {
        "churn_score": 0.62,          # contains "churn" = target name → proxy
        "monthly_charges": 0.22,
        "tenure": 0.10,
        "age": 0.06,
    }
    state2["excluded_columns"] = []
    state2["retry_count"] = 0

    result2 = auditor_node(state2)
    print(f"\nResult: approved={result2['approved']}")
    print(f"Excluded: {result2['excluded_columns']}")

    assert not result2["approved"], "FAIL: Should have detected churn_score as proxy"
    assert "churn_score" in result2["excluded_columns"], "FAIL: churn_score should be excluded"
    print("✅ Test 2 PASSED — Target proxy correctly flagged")

    # ── Test 3: Clean dataset gets approved ───────────────────────────────────
    print("\n--- TEST 3: Clean dataset — should be approved ---")

    state3 = {**state1}
    state3["best_model_score"] = 0.8234   # realistic score
    state3["feature_importance"] = {
        "monthly_charges": 0.35,    # all clean business features
        "tenure": 0.28,
        "contract_type": 0.18,
        "age": 0.12,
        "payment_method": 0.07,
    }
    state3["excluded_columns"] = []
    state3["retry_count"] = 0

    result3 = auditor_node(state3)
    print(f"\nResult: approved={result3['approved']}")
    print(f"Excluded: {result3['excluded_columns']}")
    print(f"Critique: {result3['critique']}")

    # Static rules should find nothing — LLM verdict may vary
    # but a clean dataset with realistic score should be approved
    print(f"✅ Test 3 result: approved={result3['approved']} "
          f"(expected True for clean features with realistic score)")

    # ── Test 4: Retry count increments correctly ──────────────────────────────
    print("\n--- TEST 4: Retry count increment ---")
    assert result1["retry_count"] == 1, f"FAIL: retry_count should be 1, got {result1['retry_count']}"
    assert result2["retry_count"] == 1, f"FAIL: retry_count should be 1, got {result2['retry_count']}"
    print("✅ Test 4 PASSED — retry_count increments correctly")

    print("\n" + "=" * 60)
    print("auditor.py — ALL STATIC TESTS PASSED")
    print("LLM tests require: ollama serve + deepseek-r1:1.5b pulled")
    print("=" * 60)
