# 🤖 Agentic AutoML Studio

> **An autonomous machine learning pipeline powered by DeepSeek-R1, FLAML AutoML, and LangGraph — running entirely on your local machine.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FLAML](https://img.shields.io/badge/AutoML-FLAML-orange)](https://github.com/microsoft/FLAML)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-green)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/LLM-DeepSeek--R1%3A1.5b-purple)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 What Is This?

Agentic AutoML Studio is a **multi-agent AI system** that automates the full machine learning pipeline — from raw CSV upload to a validated, leakage-free model — without any manual intervention.

Instead of a simple "load → train → show result" script, this project uses **three autonomous AI agents** that reason, act, and self-correct in a feedback loop:

```
User uploads CSV
      │
      ▼
┌─────────────┐
│ Orchestrator│  ← DeepSeek-R1 reasons: "Target = Churn, Metric = AUC"
│   (LLM)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Engineer  │  ← FLAML trains 10+ models, returns leaderboard
│  (FLAML)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Auditor   │  ← DeepSeek-R1 checks: "Is customer_id causing data leakage?"
│   (LLM)     │
└──────┬──────┘
       │
  ┌────┴────┐
  │ Approved?│
  └────┬────┘
    NO │  → Drop leaked columns → Loop back to Engineer
   YES │  → Show results in Streamlit dashboard
```

This is the difference between a **script** and an **agent**.

---

## ✨ Key Features

- **Autonomous target detection** — The LLM reads column names and types to decide what to predict
- **Automated model selection** — FLAML trains and ranks Random Forest, XGBoost, LightGBM, and more
- **Data leakage detection** — The Auditor agent catches ID columns, proxy features, and post-event data
- **Self-healing loop** — If leakage is found, the system drops bad columns and retrains automatically
- **Feature importance visualization** — Instant bar chart of what drives model predictions
- **Runs fully offline** — No cloud APIs, no billing, no internet required after setup
- **Hardware-efficient** — Designed for laptops with 8GB RAM, no GPU required

---

## 🧠 Architecture

```
Agentic-AutoML-Studio/
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py          # Shared memory (TypedDict) passed between agents
│   │   ├── orchestrator.py   # LLM agent: plans the ML task
│   │   ├── engineer.py       # FLAML agent: trains the models
│   │   ├── auditor.py        # LLM agent: checks for data leakage
│   │   └── graph.py          # LangGraph: wires the agents + conditional loop
│   └── ui/
│       └── dashboard.py      # Streamlit frontend
├── data/                     # Put your test CSVs here
├── requirements.txt
├── main.py                   # Entry point
└── README.md
```

### The Agent Roles

| Agent | Brain | Role |
|-------|-------|------|
| **Orchestrator** | DeepSeek-R1:1.5b | Reads schema, decides target column, metric, task type |
| **Engineer** | FLAML AutoML | Trains multiple models, returns leaderboard + feature importance |
| **Auditor** | DeepSeek-R1:1.5b | Detects data leakage in top features, triggers retraining if needed |

### Why This Stack?

| Tool | Why Chosen |
|------|-----------|
| **FLAML** (Microsoft) | Pure Python, 1.5GB RAM peak, outperforms H2O and AutoSklearn under time budgets |
| **DeepSeek-R1:1.5b** | Reasoning model, 1.1GB, runs on CPU, available via Ollama |
| **LangGraph** | Enables stateful loops — agents can cycle back, unlike a simple pipeline |
| **Streamlit** | Rapid UI, no frontend expertise needed |
| **No Docker** | Saves 300-500MB RAM — critical on 8GB machines |

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | 4 threads | 8+ threads |
| Disk | 5 GB free | 10 GB free |
| GPU | Not required | Speeds up Ollama |
| OS | Linux / macOS | Linux Mint / Ubuntu |
| Python | 3.10+ | 3.11 |

> **Tested on:** i5-7th Gen (U series), 8GB RAM, Linux Mint — fully functional with `time_limit=60s`

---

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/agentic-automl-studio.git
cd agentic-automl-studio
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ First install takes 5-10 minutes. FLAML pulls scikit-learn, XGBoost, LightGBM automatically.

### Step 4: Install Ollama and Pull the Model

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull DeepSeek-R1 1.5B (1.1 GB download)
ollama pull deepseek-r1:1.5b

# Start Ollama server (keep this running in a separate terminal)
ollama serve
```

### Step 5: Run the App

```bash
streamlit run app/ui/dashboard.py
```

Open your browser at `http://localhost:8501` — upload a CSV and click **Run**.

---

## 📋 Requirements

```
flaml[automl]==2.3.0
langgraph==0.2.28
langchain-ollama==0.2.0
langchain-core==0.3.0
streamlit==1.40.0
pandas==2.2.0
scikit-learn==1.5.0
psutil==6.1.0
```

---

## 🧪 Testing the Pipeline

The repo includes three test scenarios to verify each part of the system:

```bash
# Generate test datasets
python3 tests/generate_test_data.py

# Test 1: Clean dataset — Auditor should approve on first pass
# Upload: data/test_clean.csv → predict "target"

# Test 2: Leaky dataset — Auditor should flag customer_id and retrain
# Upload: data/test_leaky.csv → predict "target"
# Expected: excluded_columns = ['customer_id', 'target_flag'], retry_count = 1

# Test 3: Regression task
# Upload: data/test_regression.csv → predict "price"
# Expected: task_type = "regression", metric = "rmse"
```

---

## 📊 What You See in the Dashboard

After the pipeline runs, the Streamlit dashboard shows:

**Metrics Row:**
- Target column chosen by the Orchestrator
- Best model name (e.g., `LGBMClassifier`)
- Best validation score
- Audit status (Pass / Flagged)

**Tab 1 — Leaderboard:**
| Model | Score | Training Time |
|-------|-------|---------------|
| LGBMClassifier | 0.9821 | 12.3s |
| XGBClassifier | 0.9754 | 8.1s |
| RandomForestClassifier | 0.9612 | 6.4s |

**Tab 2 — Feature Importance:**
Bar chart of top 10 features ranked by contribution to model predictions.

**Tab 3 — Audit Report:**
- Which columns were flagged and why
- Number of retraining loops triggered
- Final approval status

---

## 🔬 Research Foundation

This project is grounded in the following AutoML research papers:

| Paper | Relevance |
|-------|-----------|
| [Efficient and Robust AutoML (Auto-Sklearn)](https://papers.neurips.cc/paper/5872) | Algorithm selection + ensemble construction under time budgets |
| [AutoML to Date and Beyond (ACM)](https://dl.acm.org/doi/abs/10.1145/3470918) | 7-tier autonomy taxonomy — this project targets Level 4+ |
| [Whither AutoML? (ACM CHI)](https://dl.acm.org/doi/abs/10.1145/3411764.3445306) | Human-AutoML partnership over full automation |
| [Trust in AutoML (ACM IUI)](https://dl.acm.org/doi/abs/10.1145/3377325.3377501) | Transparency features (leaderboard, importance) build user trust |
| [A Comparison of AutoML Tools](https://ieeexplore.ieee.org/abstract/document/9534091) | AutoGluon/FLAML benchmark — basis for tool selection |

---

## 🔄 How the Agentic Loop Works

```python
# Simplified view of graph.py
MAX_RETRIES = 2

def route_after_audit(state):
    if state["approved"]:         return "end"      # ✅ Clean model
    if state["retry_count"] >= 2: return "end"      # 🛑 Safety cap
    if not state["excluded_cols"]: return "end"     # 🤷 Nothing to fix
    return "engineer"                               # 🔄 Retrain without leaked columns
```

The loop is bounded at 2 retries — a deliberate safety design to prevent infinite loops on resource-constrained hardware.

---

## 🏗️ Extending the Project

Want to add more agents? The LangGraph state pattern makes it straightforward:

```python
# Add an Explainer agent after Auditor
workflow.add_node("explainer", explainer_node)
workflow.add_edge("auditor", "explainer")   # after approval
workflow.add_edge("explainer", END)
```

Ideas for extension:
- **Explainer agent** — generates a plain-English summary of the model's decisions
- **Feature engineer agent** — LLM suggests new derived features to improve score
- **Deployment agent** — saves the model and generates a REST API scaffold
- **Report agent** — auto-generates a PDF model card

---

## 📈 Performance on Reference Hardware

Tested on i5-7200U, 8GB RAM, Linux Mint, no GPU:

| Dataset | Rows | Cols | Training Time | RAM Peak | Best Score |
|---------|------|------|---------------|----------|------------|
| Iris (classification) | 150 | 5 | ~12s | ~3.2 GB | 0.9800 |
| Breast Cancer (binary) | 569 | 31 | ~25s | ~3.8 GB | 0.9820 |
| Titanic (binary) | 891 | 12 | ~35s | ~4.1 GB | 0.8340 |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/explainer-agent`
3. Commit your changes: `git commit -m "Add explainer agent"`
4. Push: `git push origin feature/explainer-agent`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

Built as part of an AutoML research project exploring agentic ML pipelines.

**Stack:** Python · FLAML · LangGraph · LangChain · Ollama · DeepSeek-R1 · Streamlit

---

<div align="center">

**If this helped you, give it a ⭐ — it keeps the project alive.**

</div>
