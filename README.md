# 🚀 Agentic AI MLOps Assistant

[![Tests](https://img.shields.io/badge/tests-67%2F67%20passing-brightgreen)](./tests)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-orange)](./BUILD_SUMMARY_v0.3.0.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](./docker-compose.yml)

> **Production-grade MLOps platform** with Agentic AI, Kubeflow orchestration, KServe model serving, and real-time monitoring.

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#-architecture)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)
- [📦 Deployment](#-deployment)
- [📚 Documentation](#-documentation)
- [🔧 Tech Stack](#-tech-stack)

---

## 🎯 Overview

A **custom, production-ready MLOps platform** combining:
- 🤖 **Agentic AI** - LangChain + CrewAI multi-agent orchestration
- 📦 **ML Orchestration** - Kubeflow Pipelines for distributed workflows
- 🎯 **Model Serving** - KServe with canary deployments & auto-scaling
- 📊 **Real-time Monitoring** - Prometheus metrics + Grafana dashboards
- 🔍 **RAG System** - Retrieval-augmented generation with vector embeddings
- 🔐 **Enterprise Security** - JWT authentication & model versioning
- 🐳 **Cloud Ready** - Docker, Kubernetes, GitHub Actions, AWS EC2

---

## ✨ Key Features

<div align="center">

| Feature | Component | Status |
|---------|-----------|--------|
| 🧠 **Multi-Agent Orchestration** | CrewAI + LangChain | ✅ |
| 📈 **Experiment Tracking** | MLflow Integration | ✅ |
| 🗂️ **Data Versioning** | DVC with S3/GCS | ✅ |
| 🔍 **Drift Detection** | Covariate, Label, Concept | ✅ |
| 🔑 **Authentication** | JWT Tokens | ✅ |
| 🔗 **Knowledge Retrieval** | RAG with Embeddings | ✅ |
| 🎯 **Pipeline Orchestration** | Kubeflow + Argo | ✅ |
| 🚀 **Model Serving** | KServe (sklearn, TF, PyTorch) | ✅ |
| 📊 **Metrics Collection** | Prometheus | ✅ |
| 📈 **Dashboards** | Grafana + Alerts | ✅ |
| 🧪 **Test Coverage** | 67/67 Tests | ✅ |

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic AI Layer                         │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  LangChain ReAct │  │  CrewAI Multi    │                │
│  │  Agent (Rohit)   │  │  Agent (Analyst) │                │
│  └──────────────────┘  └──────────────────┘                │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  /ask  │  /crew-ask  │  /plan  │  /execute  │  /metrics    │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌──────────────┬───────────────────┬──────────────┐
│   RAG Store  │   MLflow Tracker  │  DVC Manager │
│ (Pinecone)   │  (Experiments)    │  (Data Ver.) │
└──────────────┴───────────────────┴──────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration & Serving                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │Kubeflow Pipe │  │ KServe Model │  │ Prometheus   │     │
│  │ (Workflows)  │  │  (Inference) │  │  (Metrics)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Grafana Dashboards                       │
│  Real-time Metrics │ Drift Alerts │ Model Performance      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/JuniperRohit/ML-ops-Task.git
cd ML-ops-Task

# Install dependencies
pip install -e .

# (Optional) Install Kubernetes support
pip install -e ".[k8s]"
```

### Run Locally

**Terminal 1: Backend API**
```bash
uvicorn backend.app:app --reload --port 8030
# API available at http://localhost:8030
# Docs at http://localhost:8030/docs
```

**Terminal 2: Frontend UI**
```bash
streamlit run frontend/app.py --server.port 8601
# UI available at http://localhost:8601
```

**Terminal 3: Initialize Knowledge Base**
```bash
python load_docs.py
# Loads MLOps docs from knowledge/ folder
```

### Run Tests

```bash
# All tests (67/67)
pytest -v

# Specific test suite
pytest tests/test_monitoring_integration.py -v

# Quick run
pytest -q
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker compose up --build

# Services:
# - Backend: http://localhost:8030
# - Frontend: http://localhost:8601
```

---

## 📁 Project Structure

```
ML-ops-Task/
├── agentic_mlops/              # Core MLOps modules
│   ├── planner.py              # AgenticPlanner: workflow generation
│   ├── executor.py             # AgenticExecutor: task execution
│   ├── skills.py               # ML skills: train, evaluate, deploy
│   ├── mlflow_tracker.py        # Experiment tracking
│   ├── drift_detector.py        # Data drift detection
│   ├── dvc_manager.py           # Data versioning
│   ├── auth.py                  # JWT authentication
│   ├── kubeflow_integration.py  # Kubeflow pipelines
│   ├── kserve_integration.py    # KServe model serving
│   ├── prometheus_metrics.py    # Metrics collection
│   └── grafana_dashboards.py    # Dashboard templates
│
├── backend/                     # FastAPI Backend
│   ├── app.py                   # Main API endpoints
│   ├── app_secure.py            # Secure endpoint version
│   ├── rag.py                   # RAG engine
│   └── rohit_agent.py           # LangChain + CrewAI agents
│
├── frontend/                    # Streamlit Frontend
│   └── app.py                   # Chat UI
│
├── knowledge/                   # Static MLOps documentation
│   ├── mlflow.md                # MLflow docs
│   ├── kubeflow.md              # Kubeflow docs
│   ├── kserve.md                # KServe docs
│   └── ...
│
├── tests/                       # Test Suite (67 tests)
│   ├── test_agentic_flow.py
│   ├── test_backend.py
│   ├── test_integrations.py
│   └── test_monitoring_integration.py
│
├── docker-compose.yml           # Multi-container orchestration
├── Dockerfile                   # Backend image
├── Dockerfile.frontend          # Frontend image
├── pyproject.toml               # Dependencies & config
│
└── docs/
    ├── README.md                # This file
    ├── MONITORING_SETUP.md      # Monitoring guide
    ├── BUILD_SUMMARY_v0.3.0.md  # Release notes
    └── REQUIREMENTS_COMPARISON.md
```

---

## 🧪 Testing

All components tested with **67/67 passing tests**:

```
✅ Core MLOps (Planner, Executor, Skills)
✅ FastAPI Backend (Endpoints, RAG, Session Memory)
✅ Integrations (MLflow, DVC, JWT Auth, Drift Detection)
✅ Monitoring (Kubeflow, KServe, Prometheus, Grafana)
```

Run tests:
```bash
pytest -v                    # Verbose output
pytest -q                    # Quiet output
pytest --cov                 # Coverage report
```

---

## 📦 Deployment

### Docker Compose (Recommended for Local)

```bash
docker compose up --build
```

Services:
- **Backend**: `http://localhost:8030`
- **Frontend**: `http://localhost:8601`
- **Prometheus**: `http://localhost:9090` (if added)

### Kubernetes (Production)

```bash
# Deploy with Kubeflow
kubectl apply -f kubeflow-pipeline.yaml

# Deploy model with KServe
kubectl apply -f infer-model.yaml

# Monitor with Prometheus + Grafana
helm install prometheus prometheus-community/kube-prometheus-stack
```

### GitHub Actions CI/CD

Automatic workflows:
- ✅ Run tests on every push
- ✅ Build Docker images
- ✅ Push to AWS ECR
- ✅ Deploy to EC2 instances

See: `.github/workflows/mlops-deploy.yml`

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [MONITORING_SETUP.md](./MONITORING_SETUP.md) | Complete monitoring guide |
| [BUILD_SUMMARY_v0.3.0.md](./BUILD_SUMMARY_v0.3.0.md) | v0.3.0 release notes |
| [REQUIREMENTS_COMPARISON.md](./REQUIREMENTS_COMPARISON.md) | Course requirements mapping |
| [API Docs](http://localhost:8030/docs) | Interactive API documentation |

---

## 🔧 Tech Stack

**AI/ML**
- LangChain (0.1.0+) - Agentic framework
- CrewAI (0.1.0+) - Multi-agent orchestration
- Sentence Transformers - Embeddings
- Scikit-learn - ML models

**Backend**
- FastAPI - Modern async API
- Pydantic - Data validation
- SQLAlchemy - ORM (optional)

**Monitoring & Tracking**
- MLflow - Experiment tracking
- Prometheus - Metrics collection
- Grafana - Dashboards
- DVC - Data versioning

**Orchestration**
- Kubeflow Pipelines - Workflow orchestration
- KServe - Model serving
- Argo Workflows - Task scheduling

**Deployment**
- Docker & Docker Compose
- Kubernetes
- GitHub Actions
- AWS (ECR, EC2)

**Testing**
- Pytest - Test framework
- Coverage - Test coverage

---

## 🎓 Version History

| Version | Features | Tests |
|---------|----------|-------|
| v0.1.0 | Core MLOps (planner, executor, skills) | 10 ✅ |
| v0.2.0 | MLflow, DVC, Drift, Auth | 23 ✅ |
| **v0.3.0** | **Kubeflow, KServe, Prometheus, Grafana** | **67 ✅** |

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## 📄 License

MIT License - See [LICENSE](./LICENSE) file

---

## 📞 Support

- 📖 [Full Documentation](./MONITORING_SETUP.md)
- 🐛 [Issues](https://github.com/JuniperRohit/ML-ops-Task/issues)
- 💬 [Discussions](https://github.com/JuniperRohit/ML-ops-Task/discussions)

---

<div align="center">

**Built with ❤️ for MLOps Excellence**

⭐ If you find this helpful, please star the repository!

</div>

- `agentic_mlops/`
  - `config.py`: path constants and directories
  - `schemas.py`: Pydantic models for task plan and results
  - `skills.py`: Task implementations (data, training, evaluation, deployment)
  - `planner.py`: Converts objective to stepwise pipeline
  - `executor.py`: Executes tasks with root-cause, stop-on-fail
  - `api.py`: FastAPI routes for orchestrating runs
  - `cli.py`: Typer CLI for quick local use

## Quickstart

1. Install dependencies:

```bash
pip install -e .
```

2. CLI usage:

```bash
agentic-mlops plan "Train and deploy a binary classifier"
agentic-mlops execute "Train and deploy a binary classifier"
agentic-mlops serve --host 0.0.0.0 --port 8030
```

3. API usage:

```bash
curl -X POST "http://127.0.0.1:8030/execute" -H "Content-Type: application/json" -d '{"objective":"Train and deploy a model"}'
```

## Differences from common public solutions

1. Multi-tenant planning is replaced by an object-oriented, heuristics-based planner with a deterministic fallback.
2. Task skill registration is runtime-extensible via `SkillRegistry`, more testable than fixed `if/else` pipelines.
3. Execution stops on first non-success result and returns detailed step-level metadata.
4. Includes `deploy_model` simulation writing JSON artifact details rather than no-op.

## Development

- Start server: `agentic-mlops serve`
- Run in one shot: `agentic-mlops execute "Build a model"`

## Notes

- Adapt the `skills.py` methods to integrate actual cloud storage or model registry.
- Add a real LLM planning component by extending `planner.py` while keeping discrete step representation.
