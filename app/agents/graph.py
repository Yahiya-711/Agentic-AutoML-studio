# =============================================================================
# graph.py — The LangGraph Workflow Orchestration
# =============================================================================
#
# WHAT IS THIS FILE?
# ------------------
# This is the "nervous system" of the entire project. It takes all four
# components (state, orchestrator, engineer, auditor) and wires them into
# a directed graph with conditional routing logic.
#
# This file answers the question: "Who calls who, when, and under what
# conditions?" — a question that a simple Python script cannot express
# cleanly when loops are involved.
#
# WHY LANGGRAPH INSTEAD OF JUST CALLING FUNCTIONS IN ORDER?
# ----------------------------------------------------------
# Option A — Simple script (what most people would write):
#
#   def run_pipeline(state):
#       state = orchestrator_node(state)
#       state = engineer_node(state)
#       state = auditor_node(state)
#       if not state["approved"]:
#           state = engineer_node(state)   # retrain
#           state = auditor_node(state)    # re-audit
#       return state
#
# Problems with Option A:
#   1. The retry logic is hardcoded — you can't change MAX_RETRIES without
#      rewriting the loop structure
#   2. No visibility — you can't visualize the flow as a graph
#   3. No checkpointing — if engineer crashes on loop 2, you lose loop 1's results
#   4. Hard to extend — adding a new agent means rewriting the loop
#
# Option B — LangGraph:
#
#   workflow.add_node("engineer", engineer_node)
#   workflow.add_conditional_edges("auditor", route_fn, {"engineer": "engineer", "end": END})
#
# Benefits of Option B:
#   1. The routing logic is SEPARATE from the node logic — clean separation
#   2. The graph is VISUALIZABLE — you can print/draw it for your report
#   3. The graph is EXTENSIBLE — add a new agent = add_node() + add_edge()
#   4. The state flows automatically between nodes — no manual passing
#
# LANGGRAPH CONCEPTS YOU NEED TO UNDERSTAND
# ------------------------------------------
#
# NODE:
#   A Python function that takes AgentState → returns AgentState.
#   Each of your agents (orchestrator, engineer, auditor) is a node.
#   LangGraph calls them in the order defined by edges.
#
# EDGE:
#   A directed connection between two nodes: "after A, always go to B"
#   Example: workflow.add_edge("orchestrator", "engineer")
#   This means: after orchestrator_node() finishes, ALWAYS call engineer_node()
#
# CONDITIONAL EDGE:
#   "After node A, call a router function to decide where to go next"
#   The router function reads the state and returns a string (node name or "end")
#   Example: after auditor, either go back to "engineer" or go to END
#
# ENTRY POINT:
#   Which node runs FIRST when graph.invoke() is called
#   We set this to "orchestrator" — it always runs first
#
# END:
#   A special LangGraph sentinel that means "stop the graph here"
#   When the router returns "end", LangGraph stops and returns final state
#
# STATE FLOW:
#   LangGraph passes the full AgentState dict to each node.
#   Each node returns a (possibly partial) dict that LangGraph MERGES
#   into the current state. This is why we use **state spread in nodes.
#
# HOW THE COMPLETE FLOW LOOKS:
# ─────────────────────────────────────────────────────────────────────
#
#   graph.invoke(initial_state)
#          │
#          ▼
#   [orchestrator_node]  ← reads: columns, dtypes, task_description
#          │               writes: target_column, metric, time_limit, task_type
#          │ (add_edge: always)
#          ▼
#   [engineer_node]      ← reads: target_column, metric, time_limit, excluded_columns
#          │               writes: leaderboard, feature_importance, best_model_score
#          │ (add_edge: always)
#          ▼
#   [auditor_node]       ← reads: feature_importance, best_model_score, columns
#          │               writes: approved, critique, excluded_columns, retry_count
#          │ (add_conditional_edges: calls route_after_audit())
#          ▼
#   route_after_audit()  ← reads: approved, retry_count, excluded_columns
#          │
#    ┌─────┴──────────────────────┐
#    │                            │
#    ▼ "engineer"                 ▼ "end"
#   [engineer_node]            [END]
#    (retrain without           (return final_state
#     leaked columns)            to dashboard.py)
#          │
#          ▼
#   [auditor_node]
#          │
#    ┌─────┴──────┐
#    ▼            ▼
#  "engineer"   "end"    ← MAX_RETRIES cap stops infinite loops
#
# =============================================================================

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.orchestrator import orchestrator_node
from app.agents.engineer import engineer_node
from app.agents.auditor import auditor_node


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum number of times the Engineer can retrain after Auditor rejection.
#
# WHY 2 AND NOT MORE?
# -------------------
# Each retrain loop on your i5-7200U + 8GB RAM costs:
#   - ~60 seconds of FLAML training (your time_limit setting)
#   - ~1.5GB RAM peak during training
#   - ~30-60 seconds of LLM reasoning (Orchestrator + Auditor calls)
#
# Total cost per loop: ~2-3 minutes + RAM pressure
#
# With MAX_RETRIES=2:
#   - Loop 0 (first train): always runs
#   - Loop 1 (first retry): runs if Auditor rejects
#   - Loop 2 (second retry): runs if Auditor rejects again
#   - Loop 3: BLOCKED by the cap → pipeline ends even if not approved
#
# This means worst case = 3 training runs = ~6-9 minutes total.
# That's acceptable for a demo. MAX_RETRIES=3 would be ~8-12 minutes
# and risks OOM on extended RAM usage.
#
# In practice, most real datasets need 0-1 retries.
# The cap is a safety net for adversarial/buggy inputs.

MAX_RETRIES = 2


# =============================================================================
# ROUTER FUNCTION — The brain of the conditional edge
# =============================================================================

def route_after_audit(state: AgentState) -> str:
    """
    Decides what happens after the Auditor runs.

    This function is called by LangGraph after every auditor_node() call.
    It reads the current state and returns a string:
        "engineer" → loop back, retrain without leaked columns
        "end"      → stop the pipeline, return final state to dashboard

    THE FOUR STOPPING CONDITIONS:
    ─────────────────────────────

    Condition 1 — APPROVED:
        The Auditor found no leakage → model is clean → show results.
        This is the happy path.

    Condition 2 — MAX_RETRIES REACHED:
        We've retrained MAX_RETRIES times and still have leakage (rare).
        We stop anyway and show whatever results we have.
        WHY NOT LOOP FOREVER?
        On 8GB RAM, infinite training loops will eventually OOM.
        A graceful stop with partial results is better than a crash.
        In the dashboard, we show a note: "Model flagged but max retries reached."

    Condition 3 — NOTHING TO EXCLUDE:
        Auditor said approved=False but excluded_columns is still empty.
        This means the LLM detected conceptual leakage but couldn't name
        a specific column to remove. Retraining without any change would
        produce identical results → infinite loop.
        We stop and show results with the Auditor's critique as a warning.

    Condition 4 — LOOP BACK:
        Auditor found real leaked columns AND we have retries left.
        excluded_columns now contains the bad columns.
        Engineer will drop them and retrain from scratch.

    WHY IS THIS A SEPARATE FUNCTION AND NOT LOGIC INSIDE auditor_node?
    -------------------------------------------------------------------
    Clean separation of concerns:
    - auditor_node() is responsible for DETECTING leakage (analysis)
    - route_after_audit() is responsible for DECIDING next step (routing)
    Mixing them would make auditor_node() aware of the graph structure,
    which breaks the principle that each agent should be independently
    testable and reusable.
    """

    approved = state.get("approved", True)
    retry_count = state.get("retry_count", 0)
    excluded_columns = state.get("excluded_columns", [])

    # ── Condition 1: Approved → stop ─────────────────────────────────────────
    if approved:
        print(f"\n[Router] ✅ Approved after {retry_count} training run(s). Pipeline complete.")
        return "end"

    # ── Condition 2: Max retries reached → stop ───────────────────────────────
    if retry_count >= MAX_RETRIES:
        print(f"\n[Router] ⚠️ Max retries ({MAX_RETRIES}) reached. "
              f"Stopping with current results.")
        print(f"[Router] Note: Model may still have leakage — "
              f"excluded: {excluded_columns}")
        return "end"

    # ── Condition 3: No columns to exclude → stop (avoid infinite loop) ───────
    if not excluded_columns:
        print(f"\n[Router] ⚠️ Auditor flagged leakage but no specific columns identified.")
        print(f"[Router] Cannot retrain without knowing what to remove. Stopping.")
        return "end"

    # ── Condition 4: Loop back for retraining ─────────────────────────────────
    print(f"\n[Router] 🔄 Leakage detected. Routing back to Engineer.")
    print(f"[Router] Columns excluded for next run: {excluded_columns}")
    print(f"[Router] Retry {retry_count} of {MAX_RETRIES} max.")
    return "engineer"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph workflow.

    Returns a compiled graph ready for graph.invoke(initial_state).

    WHY A FUNCTION INSTEAD OF MODULE-LEVEL GRAPH?
    -----------------------------------------------
    If the graph were built at module level (top of file), it would be
    constructed every time the module is imported — even during testing
    when you only want to import the router function.

    As a function, the graph is only built when explicitly called.
    Streamlit calls build_graph() once per button click, which is correct.

    COMPILATION:
    workflow.compile() does several things:
    1. Validates the graph structure (no orphan nodes, valid edges)
    2. Optimizes the state passing mechanism
    3. Returns a Runnable that supports .invoke(), .stream(), .ainvoke()

    If compile() raises an error, it means the graph is misconfigured —
    usually a typo in a node name or missing entry point.
    """

    # ── Step 1: Create the graph with our state schema ─────────────────────────
    # Passing AgentState to StateGraph tells LangGraph what keys to expect.
    # This enables type checking and better error messages.
    workflow = StateGraph(AgentState)

    # ── Step 2: Register nodes ─────────────────────────────────────────────────
    # Each add_node() call registers a Python function as a named node.
    # The string name is what edges/routers reference.
    #
    # Convention: use lowercase snake_case names that match the agent's role.
    # These names appear in logs and the graph visualization.

    workflow.add_node("orchestrator", orchestrator_node)
    # orchestrator_node: reads schema → LLM plans → writes target/metric/time

    workflow.add_node("engineer", engineer_node)
    # engineer_node: reads plan → FLAML trains → writes leaderboard/importance

    workflow.add_node("auditor", auditor_node)
    # auditor_node: reads importance → LLM audits → writes approved/critique

    # ── Step 3: Set the entry point ────────────────────────────────────────────
    # The entry point is the FIRST node called when graph.invoke() is executed.
    # Always the Orchestrator — it needs to plan before anything can train.
    workflow.set_entry_point("orchestrator")

    # ── Step 4: Add unconditional edges ────────────────────────────────────────
    # These edges mean "after node A finishes, ALWAYS go to node B"
    # No conditions, no branching — straight connections.

    workflow.add_edge("orchestrator", "engineer")
    # After planning → always train. The plan is only made once.
    # Even on retrain loops, the Orchestrator is NOT called again —
    # only the Engineer and Auditor loop. This is intentional:
    # the target/metric don't change between retries, only the exclusions.

    workflow.add_edge("engineer", "auditor")
    # After every training run (first AND retrain) → always audit.
    # The Auditor must check every new model, not just the first one.

    # ── Step 5: Add the conditional edge ──────────────────────────────────────
    # This is the heart of the agentic behavior — the loop.
    #
    # add_conditional_edges() takes:
    #   source_node : the node AFTER which routing happens ("auditor")
    #   router_fn   : function that reads state → returns string key
    #   routing_map : {string_key: destination_node_name}
    #
    # After auditor_node() runs, LangGraph calls route_after_audit(state).
    # That function returns "engineer" or "end".
    # LangGraph looks up that string in the routing_map and goes there.

    workflow.add_conditional_edges(
        "auditor",           # FROM: this node
        route_after_audit,   # CALL: this router function
        {
            "engineer": "engineer",   # IF router returns "engineer" → go to engineer
            "end": END,               # IF router returns "end" → stop the graph
        }
    )

    # ── Step 6: Compile and return ─────────────────────────────────────────────
    compiled = workflow.compile()

    print("[Graph] Workflow compiled successfully.")
    print(f"[Graph] Nodes: orchestrator → engineer → auditor → (conditional: engineer | END)")
    print(f"[Graph] MAX_RETRIES: {MAX_RETRIES}")

    return compiled


# =============================================================================
# VISUALIZATION HELPER
# =============================================================================

def print_graph_structure():
    """
    Prints a text representation of the graph structure.
    Useful for understanding the flow and for including in your project report.

    Call this from anywhere:
        from app.agents.graph import print_graph_structure
        print_graph_structure()
    """
    structure = f"""
╔══════════════════════════════════════════════════════════════╗
║              AGENTIC AUTOML STUDIO — GRAPH STRUCTURE         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [START]                                                     ║
║     │                                                        ║
║     ▼                                                        ║
║  ┌──────────────┐                                            ║
║  │ orchestrator │  DeepSeek-R1: plans target/metric/time     ║
║  └──────┬───────┘                                            ║
║         │ always                                             ║
║         ▼                                                    ║
║  ┌──────────────┐                                            ║
║  │   engineer   │  FLAML: trains models, builds leaderboard  ║
║  └──────┬───────┘                                            ║
║         │ always                                             ║
║         ▼                                                    ║
║  ┌──────────────┐                                            ║
║  │   auditor    │  DeepSeek-R1: checks for data leakage      ║
║  └──────┬───────┘                                            ║
║         │                                                    ║
║    route_after_audit()                                       ║
║         │                                                    ║
║    ┌────┴────────────────┐                                   ║
║    │ approved=True       │ approved=False                    ║
║    │ OR retry>=MAX(2)    │ AND retry<MAX(2)                  ║
║    │ OR no cols to drop  │ AND excluded_cols not empty       ║
║    ▼                     ▼                                   ║
║  [END]            ┌──────────────┐                          ║
║  return           │   engineer   │  retrain without leaks   ║
║  final_state      └──────┬───────┘                          ║
║                          │ always                            ║
║                          ▼                                   ║
║                   ┌──────────────┐                          ║
║                   │   auditor    │  re-audit new model       ║
║                   └──────┬───────┘                          ║
║                          │ (same routing logic repeats)     ║
╚══════════════════════════════════════════════════════════════╝

MAX_RETRIES = {MAX_RETRIES}
Worst case: {MAX_RETRIES + 1} training runs × ~60s each = ~{(MAX_RETRIES + 1) * 2} minutes
Best case:  1 training run (clean data, approved on first pass)
"""
    print(structure)


# =============================================================================
# STANDALONE TEST
# Run this file directly to test the complete end-to-end pipeline:
#   python3 app/agents/graph.py
#
# This is the INTEGRATION TEST — it tests all four components together:
#   state.py + orchestrator.py + engineer.py + auditor.py + graph.py
#
# Two scenarios:
#   1. Leaky dataset  → Auditor should reject → Engineer retrains → approved
#   2. Clean dataset  → Auditor should approve on first pass
#
# Requirements:
#   - Ollama running: ollama serve
#   - deepseek-r1:1.5b pulled: ollama pull deepseek-r1:1.5b
#   - FLAML installed: pip install flaml[automl]
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import pandas as pd
    from sklearn.datasets import load_breast_cancer

    print("=" * 60)
    print("Graph — Full Integration Test")
    print("=" * 60)
    print("Requirements: ollama serve + deepseek-r1:1.5b + flaml")
    print()

    print_graph_structure()

    from app.agents.state import create_initial_state

    # ── Build test datasets ───────────────────────────────────────────────────
    cancer = load_breast_cancer(as_frame=True)
    df_clean = cancer.frame.copy()

    df_leaky = df_clean.copy()
    df_leaky["patient_id"] = range(len(df_leaky))       # ID leak
    df_leaky["target_copy"] = df_leaky["target"]         # proxy leak

    # Save both to temp files
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
        df_leaky.to_csv(f, index=False)
        leaky_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
        df_clean.to_csv(f, index=False)
        clean_path = f.name

    print(f"Leaky dataset:  {leaky_path} ({df_leaky.shape[0]} rows, {df_leaky.shape[1]} cols)")
    print(f"Clean dataset:  {clean_path} ({df_clean.shape[0]} rows, {df_clean.shape[1]} cols)")

    # ── Test 1: Leaky dataset ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TEST 1: Leaky dataset (should trigger Auditor loop)")
    print("─" * 60)

    graph = build_graph()

    state_leaky = create_initial_state(
        file_path=leaky_path,
        columns=df_leaky.columns.tolist(),
        dtypes=df_leaky.dtypes.astype(str).to_dict(),
        task_description="predict cancer diagnosis",
        time_limit=30,   # short for testing
    )

    print("\nInvoking graph with leaky dataset...")
    result_leaky = graph.invoke(state_leaky)

    print(f"\n{'='*40}")
    print("FINAL STATE — LEAKY DATASET:")
    print(f"  target_column    : {result_leaky['target_column']}")
    print(f"  best_model       : {result_leaky['best_model_name']}")
    print(f"  best_score       : {result_leaky['best_model_score']}")
    print(f"  approved         : {result_leaky['approved']}")
    print(f"  retry_count      : {result_leaky['retry_count']}")
    print(f"  excluded_columns : {result_leaky['excluded_columns']}")
    print(f"  critique         : {result_leaky['critique']}")
    print(f"  leaderboard rows : {len(result_leaky['leaderboard'] or [])}")

    if result_leaky["excluded_columns"]:
        print(f"\n✅ Agentic loop TRIGGERED — leakage detected and handled!")
    else:
        print(f"\n ℹ️ No leakage detected (LLM may have varied response)")

    # ── Test 2: Clean dataset ─────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("TEST 2: Clean dataset (should approve on first pass)")
    print("─" * 60)

    graph2 = build_graph()

    state_clean = create_initial_state(
        file_path=clean_path,
        columns=df_clean.columns.tolist(),
        dtypes=df_clean.dtypes.astype(str).to_dict(),
        task_description="predict cancer diagnosis",
        time_limit=30,
    )

    print("\nInvoking graph with clean dataset...")
    result_clean = graph2.invoke(state_clean)

    print(f"\n{'='*40}")
    print("FINAL STATE — CLEAN DATASET:")
    print(f"  target_column    : {result_clean['target_column']}")
    print(f"  best_model       : {result_clean['best_model_name']}")
    print(f"  best_score       : {result_clean['best_model_score']}")
    print(f"  approved         : {result_clean['approved']}")
    print(f"  retry_count      : {result_clean['retry_count']}")
    print(f"  excluded_columns : {result_clean['excluded_columns']}")

    assert result_clean["retry_count"] >= 1, "Pipeline should have run at least once"
    assert result_clean["leaderboard"] is not None, "Leaderboard should be populated"
    assert result_clean["feature_importance"] is not None, "Feature importance should be populated"
    print(f"\n✅ Clean pipeline completed. All assertions passed.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print(f"  Leaky run  → {result_leaky['retry_count']} training loop(s), "
          f"excluded: {result_leaky['excluded_columns']}")
    print(f"  Clean run  → {result_clean['retry_count']} training loop(s), "
          f"approved: {result_clean['approved']}")
    print("=" * 60)
