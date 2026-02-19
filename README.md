# 🤖 Agentic AutoML Studio

> **The world's first privacy-first, agentic AutoML system delivered as a portable Docker container — your data never leaves your machine.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FLAML](https://img.shields.io/badge/AutoML-FLAML%20(Microsoft)-orange)](https://github.com/microsoft/FLAML)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-green)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/LLM-DeepSeek--R1%20(Local)-purple)](https://ollama.com)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Privacy](https://img.shields.io/badge/Data%20Privacy-100%25%20Local-brightgreen)]()

---

## 🎯 The Problem This Solves

Every major AutoML platform today — Google AutoML, AWS SageMaker, Azure AutoML, H2O Cloud — has one thing in common: **your data leaves your machine and travels to their servers.**

This is not just a privacy preference. It is a legal problem:

- **GDPR** (EU): Fines of up to €20M per violation. €5.65 billion issued since 2018.
- **HIPAA** (US Healthcare): Patient records cannot leave hospital networks.
- **US CLOUD Act**: Requires US cloud providers to hand data to authorities on demand — even data stored abroad belonging to foreign citizens.
- **India DPDP Act, Brazil LGPD, China PIPL**: Each mandates local data storage with strict cross-border transfer restrictions.
- **Gartner forecast**: By 2027, 70% of enterprises adopting generative AI will consider digital sovereignty a top concern when selecting a provider.

**Agentic AutoML Studio is the answer**: a fully self-contained, AI-powered AutoML system that runs entirely on your hardware. No cloud. No data transfer. No compliance risk. Pull the Docker image and own your ML pipeline.

---

## ✨ What Makes This Unique

### The Core Innovation: Agentic Architecture + Local AI

Every other AutoML tool is a **fixed pipeline** — a deterministic sequence of steps with no reasoning.

This project introduces **three autonomous AI agents** powered by a local open-source LLM (DeepSeek-R1 via Ollama) that reason, act, and self-correct:

```
Standard AutoML:
  Load → Train → Show Result
  (if model is broken: YOU figure it out)

Agentic AutoML Studio:
  Load → Agent Plans → Agent Trains → Agent Audits →
  [if data leakage found: Agent names bad columns + retrains] →
  Show verified, clean results
```

The **Auditor agent** is the first LLM-powered automated data leakage detector integrated into an AutoML pipeline. It catches the silent killer of ML models — ID columns, target proxies, future data — automatically.

### vs. Every Other AutoML Framework

| Feature | Google AutoML | H2O Cloud | AutoSklearn | AutoGluon | **This Project** |
|---------|:------------:|:---------:|:-----------:|:---------:|:---------------:|
| Data stays local | ❌ | ❌ | ✅ | ✅ | ✅ **Guaranteed** |
| Agentic AI reasoning | ❌ | ❌ | ❌ | ❌ | ✅ **3 LLM Agents** |
| Auto leakage detection | ❌ | ❌ | ❌ | ❌ | ✅ **Auditor Agent** |
| Self-correcting retrain | ❌ | ❌ | ❌ | ❌ | ✅ **Auto-loop** |
| Open-source local LLM | ❌ | ❌ | ❌ | ❌ | ✅ **DeepSeek-R1** |
| Docker BaaS deployment | ❌ | ❌ | ❌ | ❌ | ✅ **One command** |
| GPU required | No | No | No | Recommended | ✅ **No** |
| GDPR / HIPAA safe | ❌ | ❌ | ✅ | ✅ | ✅ **By design** |
| Cost per run | $$$  | $$$ | Free | Free | ✅ **Free** |

---

## 🏗️ Architecture

### Three-Agent System

```
User uploads CSV
      │
      ▼
╔═════════════════════════════════════════════════╗
║            AGENTIC AUTOML STUDIO                ║
║                                                 ║
║  ┌──────────────────────────────────────────┐   ║
║  │  🧠 ORCHESTRATOR AGENT (DeepSeek-R1)     │   ║
║  │                                          │   ║
║  │  Reads column names + data types         │   ║
║  │  Understands user's business goal        │   ║
║  │  Decides: target column, metric,         │   ║
║  │  task type (classification/regression)   │   ║
║  └──────────────────┬───────────────────────┘   ║
║                     │ Structured Plan            ║
║                     ▼                           ║
║  ┌──────────────────────────────────────────┐   ║
║  │  ⚙️  ENGINEER AGENT (FLAML AutoML)       │   ║
║  │                                          │   ║
║  │  Trains LightGBM, XGBoost, RandomForest  │   ║
║  │  CatBoost, LogisticRegression + more     │   ║
║  │  Returns ranked leaderboard +            │   ║
║  │  feature importance scores               │   ║
║  └──────────────────┬───────────────────────┘   ║
║                     │ Results                    ║
║                     ▼                           ║
║  ┌──────────────────────────────────────────┐   ║
║  │  🔍 AUDITOR AGENT (DeepSeek-R1)          │   ║
║  │                                          │   ║
║  │  Inspects top features semantically      │   ║
║  │  TYPE 1: ID columns (customer_id)        │   ║
║  │  TYPE 2: Target proxies (churn_flag)     │   ║
║  │  TYPE 3: Future data (cancel_date)       │   ║
║  │  Names bad columns → triggers retrain    │   ║
║  └──────────────────┬───────────────────────┘   ║
║                     │                           ║
║            ┌────────┴──────────┐               ║
║            ▼ Approved          ▼ Leakage Found  ║
║          [END]          Exclude bad columns      ║
║       Show results      Loop to Engineer         ║
║                         (max 2 retries)          ║
╚═════════════════════════════════════════════════╝
      │
      ▼
Streamlit Dashboard:
• Model Leaderboard  • Feature Importance  • Audit Report
```

### Technology Stack Explained

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Reasoning agents** | DeepSeek-R1:1.5b via Ollama | Open-source, 1.1GB, runs on CPU, 128K context |
| **AutoML engine** | FLAML (Microsoft Research) | Outperforms H2O/AutoSklearn under time budgets at 10% resource cost |
| **Agent orchestration** | LangGraph | Stateful conditional loops — impossible with simple pipelines |
| **Frontend** | Streamlit | Full web UI in pure Python, no JS required |
| **Containerization** | Docker + Compose | Portable, reproducible, one-command deploy |
| **Privacy guarantee** | 100% local stack | No external API calls, no telemetry, air-gap compatible |

---

## 🔒 Privacy-First by Technical Design

This is not a marketing claim. It is enforced at the architecture level:

**No external network calls:** Ollama serves the LLM locally. FLAML trains locally. LangChain tracing is explicitly disabled (`LANGCHAIN_TRACING_V2=false`). Streamlit telemetry is disabled.

**No data persistence:** Uploaded CSVs are written to `/tmp/` (RAM-backed tmpfs on Linux). They are never written to a database, object store, or log file.

**Air-gap compatible:** After the one-time `docker pull` and `ollama pull deepseek-r1:1.5b`, the system runs with zero internet connectivity. Suitable for classified or sensitive environments.

**Audit trail:** Every agent decision is logged locally in the dashboard's Audit Report tab. You can see exactly what the LLM reasoned, what it flagged, and why.

```
GDPR Article 25:  ✅ Privacy by design and by default
HIPAA §164.312:   ✅ No PHI transmission outside network perimeter
India DPDP Act:   ✅ Data residency guaranteed (your hardware)
US CLOUD Act:     ✅ Not applicable (no US cloud provider involved)
```

---

## 🚀 BaaS: AutoML as a Business Service

### The Vision

Pull the Docker image. Run it on your server, laptop, or air-gapped workstation. Give any business user a URL. They upload a CSV and get a verified ML model — with no data scientist, no cloud account, no compliance review needed.

```bash
# One command. Any machine. Full ML pipeline.
docker-compose up

# Your data stays on your machine.
# Your model stays on your machine.
# Your insights stay in your business.
```

### Who Needs This

| Sector | Pain Point | How This Solves It |
|--------|-----------|-------------------|
| **Hospitals / Clinics** | Patient data cannot leave network | Runs inside hospital firewall, no cloud needed |
| **Banks / Credit Unions** | Transaction data is PCI-DSS regulated | Local Docker image, zero external transmission |
| **Law Firms** | Client data is legally privileged | Air-gap compatible, no external API |
| **EU SMEs** | Cannot afford GDPR compliance risk | No cloud = no cross-border transfer = no liability |
| **Government Agencies** | Citizen data sovereignty requirements | Fully sovereign — runs on government hardware |
| **Manufacturing** | Production IP in sensor data | No cloud exposure of proprietary process data |

### Deployment Models

```
Model 1 — Developer / Researcher:
  git clone + docker-compose up
  Full control, runs on your laptop

Model 2 — Enterprise On-Premise:
  docker pull company-registry/automl-studio
  docker-compose up
  Deployed inside corporate firewall

Model 3 — Air-Gapped / Classified:
  docker save → USB → docker load
  ollama pull (pre-downloaded model weights)
  Zero internet required after initial setup

Model 4 — Multi-Tenant (Roadmap):
  Isolated namespaces per business unit
  REST API for programmatic access
  Scheduled retraining on new data
```

---

## 📦 Quick Start

### Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| RAM | 8 GB | 5GB for container + 1.5GB Ollama + OS |
| CPU | 4 threads | No GPU required |
| Disk | 10 GB free | Docker image ~3GB + model 1.1GB |
| OS | Linux / macOS | Windows via WSL2 |
| Docker | 20.10+ | `docker --version` to check |

### Step 1: Install Ollama (Host Machine — Outside Docker)

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Pull model once (1.1 GB download)
ollama pull deepseek-r1:1.5b

# Start Ollama — keep this terminal open
ollama serve
```

### Step 2: Clone and Start

```bash
git clone https://github.com/YOUR_USERNAME/agentic-automl-studio.git
cd agentic-automl-studio

# First launch: builds image (~8 mins), downloads base layers
# Subsequent launches: ~10 seconds (all cached)
docker-compose up --build
```

### Step 3: Use the Dashboard

Open **http://localhost:8501** in your browser.

1. Upload any CSV file
2. (Optional) Describe what you want to predict
3. Adjust training time in the sidebar (30–120 seconds)
4. Click **Run Agentic Pipeline**
5. Watch three agents reason, train, and self-correct in real time

### Management Commands

```bash
# Stop the container (keeps image cached for fast restart)
docker-compose down

# View live logs from agents
docker-compose logs -f

# Restart after changes to code
docker-compose up --build

# Check container health
docker ps  # STATUS should show "healthy"

# Remove everything (forces full rebuild next time)
docker-compose down --rmi all --volumes
```

---

## 🧪 Testing the Agentic Loop

```bash
# Generate test datasets
python3 -c "
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer(as_frame=True).frame

# Test 1: Clean data — Auditor should approve on first pass
cancer.to_csv('data/test_clean.csv', index=False)

# Test 2: Leaky data — Auditor should flag and trigger retrain
leaky = cancer.copy()
leaky['patient_id'] = range(len(leaky))       # TYPE 1: ID column
leaky['target_proxy'] = leaky['target'] * 0.99 # TYPE 2: target proxy
leaky.to_csv('data/test_leaky.csv', index=False)

print('Test files ready: data/test_clean.csv and data/test_leaky.csv')
"
```

Upload `test_leaky.csv` → expected behavior:
- Auditor flags `patient_id` (ID column) and `target_proxy` (proxy)
- Pipeline retrains without those columns
- Audit Report shows: `excluded_columns: ['patient_id', 'target_proxy']`, `retry_count: 1`
- Final model score will be **lower but honest** — no more cheating

---

## 📊 Performance

Benchmarked on i5-7th Gen, 8GB RAM, Linux Mint, no GPU — the most constrained realistic hardware:

| Dataset | Rows | Pipeline Time | RAM Peak | Verdict |
|---------|------|---------------|----------|---------|
| Breast Cancer (clean) | 569 | ~95s | 3.8 GB | ✅ Approved, 1 pass |
| Breast Cancer (leaky) | 569 | ~165s | 4.1 GB | ✅ Leakage caught, retrained |
| Titanic (binary) | 891 | ~110s | 3.5 GB | ✅ Approved, 1 pass |
| Iris (multiclass) | 150 | ~75s | 3.2 GB | ✅ Approved, 1 pass |

---

## 📁 Project Structure

```
Agentic-AutoML-Studio/
├── app/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py          # Shared memory (TypedDict) across agents
│   │   ├── orchestrator.py   # LLM Agent: plans ML task from schema
│   │   ├── engineer.py       # FLAML Agent: trains and ranks models
│   │   ├── auditor.py        # LLM Agent: detects leakage, triggers retrain
│   │   └── graph.py          # LangGraph: wires agents + conditional loop
│   └── ui/
│       └── dashboard.py      # Streamlit frontend
├── data/                     # Docker volume (uploaded CSVs)
├── Dockerfile                # Container build instructions
├── docker-compose.yml        # Full stack + resource limits
├── .dockerignore             # Keeps image lean (~3GB not ~5GB)
├── docker_check.py           # Docker vs native environment detection
├── main.py                   # Entry point + preflight checks
├── requirements.txt          # Pinned Python dependencies
└── README.md
```

---

## 🔬 Research Foundation

| Paper | Key Insight | How Used |
|-------|------------|----------|
| [AutoML to Date and Beyond (ACM 2021)](https://dl.acm.org/doi/abs/10.1145/3470918) | Defined 7-tier autonomy taxonomy | This project targets Level 4+ (self-correcting) |
| [Trust in AutoML (ACM IUI 2020)](https://dl.acm.org/doi/abs/10.1145/3377325.3377501) | Leaderboard + importance = highest trust | Dashboard design centered on these |
| [Whither AutoML? (ACM CHI 2021)](https://dl.acm.org/doi/abs/10.1145/3411764.3445306) | Partnership > full automation | Auditor provides human-like judgment |
| [FLAML (Microsoft 2021)](https://github.com/microsoft/FLAML) | 10% resource, equal/better performance | Core AutoML engine choice |
| [Auto-Sklearn (NeurIPS 2015)](https://papers.neurips.cc/paper/5872) | Algorithm selection under time budgets | Mathematical basis of Engineer agent |

---

## 🗺️ Roadmap

- [x] **v1.0** — Three-agent pipeline + Docker + Leakage detection
- [ ] **v1.1** — Explainer agent (LLM plain-English model summary)
- [ ] **v1.1** — Model export (`.pkl` download from dashboard)
- [ ] **v1.2** — REST API mode (`POST /api/train` endpoint)
- [ ] **v2.0** — Multi-tenant namespaces + model registry
- [ ] **v2.0** — Federated learning (train across sites, no data sharing)

---

## 📄 License

MIT — free to use, modify, and deploy in commercial products.

---

<div align="center">

**⭐ Star this repo if you believe ML should be private by default**

*Built to prove that privacy-first AI and cutting-edge AutoML are not mutually exclusive.*

</div>
