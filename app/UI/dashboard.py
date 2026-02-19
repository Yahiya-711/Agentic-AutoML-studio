# =============================================================================
# dashboard.py — Streamlit Frontend for Agentic AutoML Studio
# =============================================================================
#
# WHAT IS THIS FILE?
# ------------------
# This is the user-facing layer of the entire project. It is the ONLY file
# the user directly interacts with. Everything else (agents, graph, state)
# runs invisibly in the background when the user clicks "Run Pipeline".
#
# Streamlit turns Python scripts into web applications. Every time the user
# interacts with a widget (slider, button, file upload), Streamlit re-runs
# this entire script from top to bottom. This is called the "rerun model".
#
# STREAMLIT'S RERUN MODEL — CRITICAL TO UNDERSTAND
# -------------------------------------------------
# Unlike Flask/Django where you define routes and handlers, Streamlit is:
#   1. User interacts → entire script reruns
#   2. Widgets remember their values via st.session_state
#   3. Expensive computations must be guarded with if-blocks or st.cache
#
# This means:
#   - The graph.invoke() call MUST be inside an if-block (the button click)
#   - Results must be stored in st.session_state to survive reruns
#   - File uploads are re-read on every rerun unless cached in session_state
#
# WHY STREAMLIT AND NOT FLASK OR FASTAPI?
# ----------------------------------------
# Flask/FastAPI require:
#   - HTML templates for UI
#   - JavaScript for interactivity
#   - CSS for styling
#   - API endpoint design
#
# Streamlit requires: Python only.
# For a project with a Sunday deadline, Streamlit is the correct choice.
# The entire dashboard is ~200 lines of pure Python.
#
# LAYOUT STRUCTURE
# ----------------
#   ┌─────────────────────────────────────────────────┐
#   │  HEADER: Title + caption + RAM status badge     │
#   ├────────────┬────────────────────────────────────┤
#   │  SIDEBAR   │  MAIN AREA                         │
#   │  Settings  │  File uploader                     │
#   │  - Time    │  Dataset preview                   │
#   │  - Stack   │  Task description input            │
#   │  - Status  │  Run button                        │
#   │            │                                    │
#   │            │  ── After pipeline runs ──         │
#   │            │  Metrics row (4 columns)           │
#   │            │  Agent trace (expander)            │
#   │            │  Tabs:                             │
#   │            │    [Leaderboard][Features][Audit]  │
#   └────────────┴────────────────────────────────────┘
#
# =============================================================================

import os
import sys
import time
import psutil
import pandas as pd
import streamlit as st

# Add project root to Python path so imports work regardless of
# where the user runs the script from (cd project/ vs cd project/app/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from app.agents.graph import build_graph, print_graph_structure
from app.agents.state import create_initial_state


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Must be the FIRST Streamlit call in the script — before any other st.* call.
# Any st.* call before set_page_config() raises an error.
#
# layout="wide": uses full browser width instead of centered narrow column.
#   On your 1366×768 laptop screen, "wide" gives you ~30% more horizontal
#   space for the leaderboard and feature importance charts.
#
# initial_sidebar_state="expanded": sidebar open by default so users see
#   the settings immediately without hunting for them.

st.set_page_config(
    page_title="Agentic AutoML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================
# =============================================================================
# CUSTOM CSS - FIXED FOR DARK THEME
# =============================================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: white !important;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
        color: #e0e0e0 !important;
    }

    /* Metric cards - FIX: Ensure text is visible */
    [data-testid="stMetric"] {
        background: #1a1f2e !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #68d391 !important;
        font-size: 0.875rem !important;
    }

    /* Status badges - with better contrast */
    .badge-green  { background:#22543d; color:#81e6d9; padding:4px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; display:inline-block; }
    .badge-yellow { background:#744210; color:#f6e05e; padding:4px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; display:inline-block; }
    .badge-red    { background:#742a2a; color:#fc8181; padding:4px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; display:inline-block; }
    .badge-blue   { background:#2c5282; color:#90cdf4; padding:4px 10px; border-radius:20px; font-size:0.85rem; font-weight:600; display:inline-block; }

    /* Agent trace box */
    .agent-trace {
        background: #0d1117;
        color: #58a6ff;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #30363d;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
    }

    /* Input fields - FIX: Text must be visible */
    input[type="text"], input[type="number"], textarea {
        color: #ffffff !important;
        background-color: #1a202c !important;
    }
    input[type="text"]::placeholder, textarea::placeholder {
        color: #718096 !important;
    }

    /* Select dropdowns */
    select {
        color: #ffffff !important;
        background-color: #1a202c !important;
    }

    /* Dataframe tables */
    .stDataFrame {
        color: #e2e8f0 !important;
    }

    /* Fix white boxes in cards */
    div[data-testid="stVerticalBlock"] {
        color: #ffffff !important;
    }

    /* Leaderboard rank highlight */
    .rank-1 { background: #ffd700; color: #000000; font-weight: bold; }

    /* Hide Streamlit branding */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ram_status() -> tuple:
    """
    Returns current RAM usage as (percent_used, free_gb, color_class).

    WHY MONITOR RAM IN THE DASHBOARD?
    On your 8GB machine, the user might have Chrome open with 20 tabs
    before running the pipeline. 85%+ RAM usage before training starts
    means the FLAML training run will hit swap and be painfully slow.
    This function shows a live RAM badge in the sidebar so the user
    knows to close apps before clicking Run.

    Returns color_class for CSS badge:
        green  → < 70% used, safe to run
        yellow → 70-85% used, close some apps
        red    → > 85% used, high risk of slowdown
    """
    mem = psutil.virtual_memory()
    percent = mem.percent
    free_gb = round(mem.available / (1024 ** 3), 1)

    if percent < 70:
        color = "badge-green"
    elif percent < 85:
        color = "badge-yellow"
    else:
        color = "badge-red"

    return percent, free_gb, color


def save_uploaded_file(uploaded_file) -> str:
    """
    Save Streamlit's UploadedFile object to /tmp/ and return the path.

    WHY SAVE TO /tmp/ AND NOT KEEP IN MEMORY?
    Streamlit's UploadedFile is an in-memory BytesIO object. The agents
    (especially engineer.py) need a file PATH to pass to pandas.read_csv()
    and FLAML. We can't pass a BytesIO object to FLAML directly.

    /tmp/ is RAM-backed on Linux (tmpfs) — reading from it is fast
    and it doesn't wear your SSD. Files there are auto-cleaned on reboot.

    WHY USE THE ORIGINAL FILENAME?
    Makes debugging easier — if the pipeline fails, the error message
    mentions "/tmp/customer_churn.csv" which is immediately recognizable.
    A UUID-named file like "/tmp/a3f2b1.csv" is harder to trace.
    """
    save_path = os.path.join("/tmp", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def format_leaderboard_df(leaderboard: list) -> pd.DataFrame:
    """
    Convert the leaderboard list-of-dicts to a formatted DataFrame.

    Adds a Rank column and formats the score as a percentage string
    for readability. The raw score (0.9821) is harder to parse at a
    glance than "98.21%" during a live demo.
    """
    if not leaderboard:
        return pd.DataFrame()

    df = pd.DataFrame(leaderboard)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Format score as percentage if it looks like a 0-1 metric
    if "score" in df.columns:
        if df["score"].max() <= 1.0:
            df["score_pct"] = df["score"].apply(lambda x: f"{x*100:.2f}%")
        else:
            df["score_pct"] = df["score"].apply(lambda x: f"{x:.4f}")

    # Rename columns for display
    col_rename = {
        "model": "Model",
        "score": "Score (raw)",
        "score_pct": "Score",
        "training_time_s": "Train Time (s)",
    }
    df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})

    return df


def format_feature_importance_df(feature_importance: dict) -> pd.DataFrame:
    """
    Convert feature importance dict to a sorted DataFrame for display.
    Adds a percentage column for the bar chart overlay.
    """
    if not feature_importance:
        return pd.DataFrame()

    df = (
        pd.DataFrame(
            feature_importance.items(),
            columns=["Feature", "Importance"]
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "Rank", range(1, len(df) + 1))
    df["Importance %"] = df["Importance"].apply(lambda x: f"{x*100:.2f}%")

    return df


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """
    Renders the settings sidebar and returns user-configured values.

    The sidebar contains:
    1. RAM status badge (live hardware check)
    2. Training time slider
    3. Tech stack info card
    4. Tips section

    Returns:
        time_limit (int): seconds for FLAML training
    """
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.divider()

        # ── RAM Status ──────────────────────────────────────────────────────
        st.markdown("**💾 System Status**")
        percent, free_gb, color = get_ram_status()
        st.markdown(
            f'<span class="{color}">RAM: {percent:.0f}% used · {free_gb} GB free</span>',
            unsafe_allow_html=True
        )

        if percent > 85:
            st.warning("⚠️ RAM is high. Close browser tabs and apps before running.")
        elif percent > 70:
            st.info("ℹ️ RAM is moderate. Pipeline will run but may be slower.")
        else:
            st.success("✅ RAM looks good. Ready to run.")

        st.divider()

        # ── Training Time Slider ─────────────────────────────────────────────
        st.markdown("**⏱️ Training Time Budget**")
        time_limit = st.slider(
            label="Seconds for FLAML to train",
            min_value=30,
            max_value=120,
            value=60,
            step=10,
            help=(
                "How many seconds FLAML gets to search for the best model. "
                "More time = more models tried = potentially better results. "
                "On your i5-7200U, 60s trains ~10 models comfortably."
            )
        )

        # Show what 60s means on their hardware
        estimated_models = max(5, min(20, time_limit // 6))
        st.caption(f"≈ {estimated_models} models evaluated on your CPU")

        st.divider()

        # ── Tech Stack Card ──────────────────────────────────────────────────
        st.markdown("**🧠 Agent Stack**")
        st.code(
            "LLM:     DeepSeek-R1:1.5b\n"
            "AutoML:  FLAML (Microsoft)\n"
            "Graph:   LangGraph\n"
            "Runtime: Ollama (local)",
            language="text"
        )

        st.divider()

        # ── Tips ─────────────────────────────────────────────────────────────
        with st.expander("💡 Tips for best results"):
            st.markdown("""
- **Dataset size**: 100–10,000 rows works best
- **Missing values**: FLAML handles them automatically
- **Target column**: Binary or multiclass works great
- **Leakage test**: Try adding a `row_id` column — watch the Auditor catch it!
- **Time limit**: 60s is sweet spot for demo speed vs quality
""")

    return time_limit


# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

def render_header():
    """Renders the main page header with title and description."""
    st.markdown("""
<div class="main-header">
    <h1>🤖 Agentic AutoML Studio</h1>
    <p>Upload a CSV → Three AI agents plan, train, and self-correct a machine learning model — fully automated.</p>
</div>
""", unsafe_allow_html=True)


def render_upload_section():
    """
    Renders file upload + dataset preview.
    Returns (uploaded_file, df_preview, save_path) or (None, None, None).
    """
    uploaded_file = st.file_uploader(
        label="📂 Upload your CSV dataset",
        type=["csv"],
        help="Upload any CSV file. The Orchestrator agent will automatically detect the target column.",
    )

    if uploaded_file is None:
        # Show example datasets suggestion when nothing is uploaded
        st.info(
            "👆 Upload a CSV to get started.  \n"
            "**Try these datasets:**  \n"
            "• [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — binary classification  \n"
            "• [House Prices](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) — regression  \n"
            "• Any Kaggle competition CSV"
        )
        return None, None, None

    # Save file and load preview
    save_path = save_uploaded_file(uploaded_file)
    df_preview = pd.read_csv(save_path)

    # Dataset overview
    with st.expander("📋 Dataset Preview", expanded=True):
        # Summary metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df_preview.shape[0]:,}")
        c2.metric("Columns", df_preview.shape[1])
        c3.metric("Missing Values", int(df_preview.isnull().sum().sum()))
        c4.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

        # Data preview table
        st.dataframe(
            df_preview.head(8),
            use_container_width=True,
            height=250,
        )

        # Column type breakdown
        numeric_cols = df_preview.select_dtypes(include='number').columns.tolist()
        categorical_cols = df_preview.select_dtypes(include='object').columns.tolist()
        st.caption(
            f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:8])}{'...' if len(numeric_cols) > 8 else ''}  |  "
            f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}"
        )

    return uploaded_file, df_preview, save_path


def render_task_input(df_preview: pd.DataFrame) -> str:
    """
    Renders the task description input and optional target column hint.
    Returns the task description string.
    """
    st.markdown("### 🎯 What do you want to predict?")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        task_description = st.text_input(
            label="Describe your prediction goal (optional)",
            placeholder="e.g., predict customer churn, classify loan default, forecast house price",
            help=(
                "The Orchestrator LLM reads this alongside your column names to decide "
                "which column to predict and which metric to optimize. "
                "Leave blank to let the LLM decide entirely from the schema."
            )
        )

    with col_right:
        # Optional: manual target column override
        # If the user KNOWS which column they want to predict, they can
        # specify it here and the Orchestrator will respect it via the
        # task_description (we inject the column name into the text)
        manual_target = st.selectbox(
            label="Or manually select target column",
            options=["(Let AI decide)"] + df_preview.columns.tolist(),
            index=0,
            help="Override the Orchestrator's choice if you know which column to predict."
        )

    # If user manually selected a target, inject it into task_description
    if manual_target != "(Let AI decide)":
        task_description = f"predict the column named '{manual_target}'. {task_description}"

    return task_description


def render_run_button(df_preview: pd.DataFrame, save_path: str,
                      task_description: str, time_limit: int):
    """
    Renders the Run button and executes the pipeline when clicked.

    WHY st.session_state?
    ---------------------
    Streamlit reruns the entire script on every interaction.
    Without session_state, clicking the Run button stores the result,
    but the next user interaction (like adjusting a slider) would
    rerun the script and lose the result — it would just disappear.

    st.session_state persists values ACROSS reruns within the same
    browser session. We store the final pipeline result there so it
    survives slider adjustments, tab clicks, etc.

    The pattern:
        if "result" not in st.session_state → first run, show upload
        if button clicked → run pipeline, store in session_state
        always render results from session_state if they exist
    """
    st.divider()

    # Preflight check: warn if RAM is high before expensive training
    percent, free_gb, _ = get_ram_status()
    if percent > 80:
        st.warning(
            f"⚠️ RAM is at {percent:.0f}%. "
            f"Consider closing other applications before running to avoid slowdowns."
        )

    # The Run button
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button(
            "🚀 Run Agentic Pipeline",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.markdown(
            f"<small>Will train for **{time_limit}s** · "
            f"Up to **{MAX_RETRIES_DISPLAY} retrain loops** if leakage detected · "
            f"Estimated time: **{time_limit + 90}–{(time_limit * 2) + 120}s total**</small>",
            unsafe_allow_html=True
        )

    if run_clicked:
        # Clear any previous results before new run
        if "pipeline_result" in st.session_state:
            del st.session_state["pipeline_result"]
        if "pipeline_log" in st.session_state:
            del st.session_state["pipeline_log"]

        # Build initial state
        initial_state = create_initial_state(
            file_path=save_path,
            columns=df_preview.columns.tolist(),
            dtypes=df_preview.dtypes.astype(str).to_dict(),
            task_description=task_description or "predict the most relevant target column",
            time_limit=time_limit,
        )

        # ── Execute Pipeline ──────────────────────────────────────────────────
        # We show a multi-stage progress bar to keep the user informed.
        # On your i5-7200U with 60s training, total pipeline time is
        # typically 90-150s. Without progress indication, users think
        # it crashed after 30s of silence.

        progress_bar = st.progress(0, text="🧠 Orchestrator is analyzing your dataset...")
        status_placeholder = st.empty()
        start_time = time.time()

        try:
            # Stage 1: Build graph (fast, ~0.1s)
            graph = build_graph()
            progress_bar.progress(10, text="🧠 Orchestrator analyzing schema with DeepSeek-R1...")
            status_placeholder.info("The Orchestrator LLM is reading your column names and deciding the ML task...")

            # Stage 2: Run the graph (this is the long one)
            # graph.invoke() is synchronous — it blocks until the full
            # pipeline (all loops) completes. During this time, Streamlit
            # shows the progress bar but can't update it mid-execution.
            # That's a Streamlit limitation with synchronous code.
            #
            # We update the progress to 20% before invoke() to show
            # "something is happening", then jump to 100% after.
            progress_bar.progress(20, text="⚙️ Engineer training models with FLAML...")
            status_placeholder.info(
                f"FLAML is training multiple models (LightGBM, XGBoost, RandomForest...) "
                f"for {time_limit} seconds. Please wait..."
            )

            final_state = graph.invoke(initial_state)

            elapsed = round(time.time() - start_time, 1)
            progress_bar.progress(100, text=f"✅ Pipeline complete in {elapsed}s!")
            status_placeholder.empty()

            # Store result in session_state for persistence across reruns
            st.session_state["pipeline_result"] = final_state
            st.session_state["pipeline_elapsed"] = elapsed

            st.balloons()

        except ConnectionError:
            progress_bar.empty()
            status_placeholder.error(
                "❌ Cannot connect to Ollama. "
                "Make sure Ollama is running: open a terminal and run `ollama serve`"
            )
            return

        except Exception as e:
            progress_bar.empty()
            status_placeholder.error(f"❌ Pipeline error: {str(e)}")
            st.exception(e)   # show full traceback for debugging
            return


def render_results(result: dict, elapsed: float):
    """
    Renders the full results section after a successful pipeline run.

    Structure:
        1. Summary metrics row (4 columns)
        2. Agent decision trace (expandable)
        3. Three-tab results panel:
           Tab 1: Model Leaderboard
           Tab 2: Feature Importance
           Tab 3: Audit Report + Pipeline JSON
    """
    st.divider()
    st.markdown("## 📊 Pipeline Results")

    # ── Metrics Row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        label="🎯 Target Column",
        value=result.get("target_column", "N/A"),
        help="The column the model was trained to predict, chosen by the Orchestrator."
    )

    m2.metric(
        label="🏆 Best Model",
        value=result.get("best_model_name", "N/A"),
        help="The highest-scoring model from FLAML's automated search."
    )

    score = result.get("best_model_score", 0)
    metric_name = result.get("metric", "score")
    m3.metric(
        label=f"📈 Best Score ({metric_name})",
        value=f"{score:.4f}",
        help=f"Validation {metric_name} of the best model. Higher is better for classification metrics."
    )

    audit_status = "✅ Approved" if result.get("approved") else "⚠️ Flagged"
    audit_delta = (
        f"{result.get('retry_count', 0)} retrain(s)"
        if result.get("retry_count", 0) > 1
        else "1st pass"
    )
    m4.metric(
        label="🔍 Audit Status",
        value=audit_status,
        delta=audit_delta,
        help="Whether the Auditor agent approved the model as free of data leakage."
    )

    # ── Agent Decision Trace ──────────────────────────────────────────────────
    with st.expander("🔎 Agent Decision Trace", expanded=False):
        st.markdown("**How the agents decided:**")

        # Orchestrator decision
        st.markdown(f"""
<div style="padding: 0.75rem; background: #e8f4f8; border-left: 4px solid #2196F3; border-radius: 4px; margin-bottom: 0.5rem;">
<strong>🧠 Orchestrator (DeepSeek-R1)</strong><br>
→ Target: <code>{result.get('target_column')}</code> &nbsp;|&nbsp;
Task: <code>{result.get('task_type')}</code> &nbsp;|&nbsp;
Metric: <code>{result.get('metric')}</code> &nbsp;|&nbsp;
Time limit: <code>{result.get('time_limit')}s</code>
</div>
""", unsafe_allow_html=True)

        # Engineer decision (show leaderboard count)
        n_models = len(result.get("leaderboard") or [])
        st.markdown(f"""
<div style="padding: 0.75rem; background: #e8f5e9; border-left: 4px solid #4CAF50; border-radius: 4px; margin-bottom: 0.5rem;">
<strong>⚙️ Engineer (FLAML)</strong><br>
→ Trained <code>{n_models}</code> models in <code>{result.get('time_limit')}s</code> &nbsp;|&nbsp;
Best: <code>{result.get('best_model_name')}</code> &nbsp;|&nbsp;
Score: <code>{result.get('best_model_score'):.4f}</code>
</div>
""", unsafe_allow_html=True)

        # Auditor decision
        excluded = result.get("excluded_columns", [])
        retries = result.get("retry_count", 1)
        audit_color = "#fff3e0" if excluded else "#e8f5e9"
        audit_border = "#FF9800" if excluded else "#4CAF50"
        st.markdown(f"""
<div style="padding: 0.75rem; background: {audit_color}; border-left: 4px solid {audit_border}; border-radius: 4px; margin-bottom: 0.5rem;">
<strong>🔍 Auditor (DeepSeek-R1)</strong><br>
→ Verdict: <code>{'APPROVED' if result.get('approved') else 'FLAGGED'}</code> &nbsp;|&nbsp;
Training loops: <code>{retries}</code> &nbsp;|&nbsp;
Excluded columns: <code>{excluded if excluded else 'none'}</code><br>
→ Reason: {result.get('critique', 'N/A')}
</div>
""", unsafe_allow_html=True)

        st.caption(f"Total pipeline time: {elapsed}s")

    # ── Three-Tab Results Panel ────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🏆 Model Leaderboard",
        "📊 Feature Importance",
        "🔍 Audit Report",
    ])

    # ── Tab 1: Leaderboard ────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Model Leaderboard")
        st.caption(
            f"FLAML trained and ranked these models in {result.get('time_limit')}s. "
            f"The best model (Rank #1) was selected for the audit."
        )

        leaderboard = result.get("leaderboard")
        if leaderboard:
            lb_df = format_leaderboard_df(leaderboard)

            # Highlight the best model row
            st.dataframe(
                lb_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Model": st.column_config.TextColumn("Model", width="large"),
                    "Score": st.column_config.TextColumn("Score", width="medium"),
                    "Score (raw)": st.column_config.NumberColumn("Score (raw)", format="%.4f"),
                    "Train Time (s)": st.column_config.NumberColumn("Train Time (s)", format="%.1f"),
                }
            )

            # Contextual score interpretation
            score_val = result.get("best_model_score", 0)
            metric = result.get("metric", "")
            st.divider()

            if metric in ("roc_auc", "accuracy"):
                if score_val > 0.95:
                    st.markdown(
                        f'<span class="badge-yellow">⚠️ Score {score_val:.4f} is very high — '
                        f'verify with the Audit Report tab</span>',
                        unsafe_allow_html=True
                    )
                elif score_val > 0.80:
                    st.markdown(
                        f'<span class="badge-green">✅ Score {score_val:.4f} is in a '
                        f'realistic range for this type of task</span>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<span class="badge-blue">ℹ️ Score {score_val:.4f} — consider more '
                        f'features or a longer training time</span>',
                        unsafe_allow_html=True
                    )
        else:
            st.warning("No leaderboard data available.")

    # ── Tab 2: Feature Importance ─────────────────────────────────────────────
    with tab2:
        st.markdown("### Feature Importance")
        st.caption(
            "Which features the best model relied on most. "
            "Suspiciously high-importance ID columns are flagged by the Auditor."
        )

        feat_imp = result.get("feature_importance")
        if feat_imp:
            fi_df = format_feature_importance_df(feat_imp)

            col_chart, col_table = st.columns([3, 2])

            with col_chart:
                # Bar chart — top 15 features
                chart_data = (
                    fi_df.head(15)
                    .set_index("Feature")["Importance"]
                )
                st.bar_chart(chart_data, height=400, use_container_width=True)

            with col_table:
                # Full sortable table
                st.dataframe(
                    fi_df[["Rank", "Feature", "Importance %"]].head(20),
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )

            # Highlight excluded columns in feature importance
            excluded = result.get("excluded_columns", [])
            if excluded:
                st.warning(
                    f"🚫 The following columns were **removed by the Auditor** "
                    f"and are NOT shown above: `{'`, `'.join(excluded)}`"
                )
        else:
            st.warning("No feature importance data available.")

    # ── Tab 3: Audit Report ───────────────────────────────────────────────────
    with tab3:
        st.markdown("### Audit Report")

        excluded = result.get("excluded_columns", [])
        approved = result.get("approved", True)
        retries = result.get("retry_count", 1)

        if approved and not excluded:
            st.success(
                "✅ **Model Approved** — The Auditor found no data leakage in the top features. "
                "The model learned from legitimate business features only."
            )
        elif approved and excluded:
            st.warning(
                f"✅ **Model Approved After Cleanup** — The Auditor detected and removed "
                f"`{'`, `'.join(excluded)}` "
                f"in a previous training loop. The final model is clean."
            )
        else:
            st.error(
                f"⚠️ **Model Flagged** — The Auditor detected potential leakage but max retries "
                f"({MAX_RETRIES_DISPLAY}) were reached. Use results with caution."
            )

        # Audit details
        st.markdown("#### Auditor's Assessment")
        st.info(f"**Verdict:** {result.get('critique', 'No critique available.')}")

        if excluded:
            st.markdown("#### Columns Removed During Pipeline")
            for col in excluded:
                st.markdown(f"- 🚫 `{col}` — flagged as potential data leakage")

        st.markdown("#### Pipeline Statistics")
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Training Loops", retries)
        stat2.metric("Columns Removed", len(excluded))
        stat3.metric("Total Time", f"{elapsed}s")

        # Full pipeline state as JSON (for debugging/report)
        with st.expander("🔧 Full Pipeline State (JSON)", expanded=False):
            st.caption("Complete state dict returned by LangGraph — useful for debugging.")
            # Show a clean subset (skip raw leaderboard/importance dicts for readability)
            display_state = {
                "target_column": result.get("target_column"),
                "task_type": result.get("task_type"),
                "metric": result.get("metric"),
                "time_limit": result.get("time_limit"),
                "best_model_name": result.get("best_model_name"),
                "best_model_score": result.get("best_model_score"),
                "approved": result.get("approved"),
                "critique": result.get("critique"),
                "excluded_columns": result.get("excluded_columns"),
                "retry_count": result.get("retry_count"),
                "leaderboard_rows": len(result.get("leaderboard") or []),
                "features_tracked": len(result.get("feature_importance") or {}),
            }
            st.json(display_state)


# =============================================================================
# DISPLAY CONSTANT (used in render_run_button without importing from graph.py)
# =============================================================================
MAX_RETRIES_DISPLAY = 2


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main function — orchestrates the entire Streamlit page.

    Execution order on every page load/rerun:
    1. Render header (always)
    2. Render sidebar (always) → get time_limit setting
    3. Render file upload section → get df_preview, save_path
    4. If file uploaded:
       a. Render task input → get task_description
       b. Render run button → triggers pipeline on click
       c. If session_state has result → render results
    """
    render_header()

    # Sidebar renders on every page load — returns user settings
    time_limit = render_sidebar()

    # File upload section
    uploaded_file, df_preview, save_path = render_upload_section()

    if uploaded_file is None:
        # Nothing uploaded yet — show welcome message
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
**🧠 Step 1: Orchestrator**
The DeepSeek-R1 LLM reads your column names and decides which column to predict, which metric to optimize, and what type of task this is.
""")
        with col2:
            st.markdown("""
**⚙️ Step 2: Engineer**
FLAML AutoML trains 10+ models (LightGBM, XGBoost, RandomForest, etc.) and ranks them by performance in your time budget.
""")
        with col3:
            st.markdown("""
**🔍 Step 3: Auditor**
DeepSeek-R1 inspects the top features for data leakage. If found, it names the bad columns, triggers a retrain, and checks again.
""")
        return

    # Task description input
    task_description = render_task_input(df_preview)

    # Run button — triggers pipeline if clicked, stores result in session_state
    render_run_button(df_preview, save_path, task_description, time_limit)

    # Results — rendered from session_state if pipeline has run
    if "pipeline_result" in st.session_state:
        render_results(
            result=st.session_state["pipeline_result"],
            elapsed=st.session_state.get("pipeline_elapsed", 0),
        )


# =============================================================================
# ENTRY POINT
# =============================================================================
# Streamlit runs this file as a script, not as a module.
# The if __name__ == "__main__" guard is good practice but Streamlit
# would execute main() either way. We include it for clarity and so
# the file is also directly runnable with `python3 dashboard.py`.

if __name__ == "__main__":
    main()
