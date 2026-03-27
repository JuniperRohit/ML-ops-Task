# Integration Features - v0.2.0

This document describes the new features added to meet MLOps course requirements.

## 🆕 NEW FEATURES & INTEGRATIONS

### 1. MLflow Experiment Tracking
**Status**: ✅ INTEGRATED

MLflow tracking has been integrated into the ML training pipeline for comprehensive experiment management.

**Features**:
- Automatic experiment logging
- Parameter tracking (penalty, C value, sample size, features)
- Metrics logging (accuracy, drift scores)
- Model artifact storage
- Run history management

**Usage**:
```python
from agentic_mlops.mlflow_tracker import get_mlflow_tracker

mlflow = get_mlflow_tracker()
mlflow.start_run(run_name="my_experiment")
mlflow.log_params({"learning_rate": 0.01})
mlflow.log_metrics({"accuracy": 0.95})
mlflow.end_run()
```

**Integration Points**:
- `agentic_mlops/mlflow_tracker.py` - Core MLflow wrapper
- `agentic_mlops/skills.py` - Training & evaluation with MLflow logging
- `mlruns/` - Local MLflow runs directory

**View Results**:
```bash
mlflow ui  # Open at http://localhost:5000
```

---

### 2. Model Drift Detection
**Status**: ✅ INTEGRATED

Automated detection of data and model drift with statistical analysis.

**Types of Drift Detected**:
- **Covariate Shift**: Input feature distribution change
- **Label Shift**: Output label distribution change
- **Concept Shift**: Decision boundary change

**Features**:
- Baseline data comparison
- Multiple drift detection methods
- Risk scoring (0.0-1.0)
- Detailed drift metrics

**Usage**:
```python
from agentic_mlops.drift_detector import DriftDetector

detector = DriftDetector(threshold=0.3)
detector.set_baseline(X_train, y_train)

metrics = detector.check_drift(X_new, check_type="covariate")
print(f"Detected: {metrics.detected}, Score: {metrics.score}")
```

**Integration Points**:
- `agentic_mlops/drift_detector.py` - Drift detection engine
- `agentic_mlops/skills.py` - Automatic drift checks during evaluation
- Logged to MLflow as metrics

---

### 3. Data Versioning with DVC
**Status**: ✅ INTEGRATED

Data Version Control (DVC) support for tracking data changes with Git-like semantics.

**Features**:
- Data tracking & versioning
- Remote storage support (S3, GCS, local)
- Pipeline definition
- Dependency management
- Data reproducibility

**Usage**:
```python
from agentic_mlops.dvc_manager import DVCManager, setup_dvc_s3

# Initialize DVC
manager = DVCManager()
manager.init()

# Setup S3 remote
setup_dvc_s3("my-bucket")

# Add data files
manager.add_data("data/generated_data.npz")
manager.push_data()
```

**Commands**:
```bash
dvc init              # Initialize DVC
dvc add data.csv      # Track data file
dvc push              # Push to remote
dvc pull              # Pull from remote
dvc repro            # Reproduce pipeline
```

**Integration Points**:
- `agentic_mlops/dvc_manager.py` - DVC wrapper
- Optional: `.dvc/config` - DVC configuration
- Optional: `dvc.yaml` - Pipeline definition

---

### 4. JWT Authentication & Security
**Status**: ✅ INTEGRATED

Secure API access with JWT token-based authentication.

**Features**:
- User token generation
- Token validation & verification
- Expiration handling
- Secure credentials
- Optional authentication (public + private endpoints)

**Usage**:
```python
from agentic_mlops.auth import create_user_token, verify_token

# Create token
token = create_user_token("username", "email@example.com")

# Use in API request
headers = {"Authorization": f"Bearer {token.access_token}"}

# Verify token
user = verify_token(token.access_token)
```

**API Endpoint**: `/login`
```bash
curl -X POST http://localhost:8030/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Returns:
# {"access_token": "eyJ...", "token_type": "bearer", "expires_in": 1800}
```

**Protected Endpoints**:
- `/metrics` - Requires authentication
- `/ask`, `/crew-ask` - Optional authentication

**Integration Points**:
- `agentic_mlops/auth.py` - JWT implementation
- `backend/app_secure.py` - Secure FastAPI endpoints
- `.env` - SECRET_KEY configuration

---

### 5. Enhanced API with Security
**Status**: ✅ INTEGRATED

New secure API server with authentication support.

**New Endpoints**:
- `POST /login` - Get JWT token
- `GET /metrics` - System metrics (requires auth)
- `POST /ask` - Ask questions (optional auth)
- `POST /crew-ask` - Multi-agent Q&A (optional auth)

**Features**:
- Bearer token authentication
- Optional authentication (mixed public/private)
- Session tracking
- Metrics collection
- Error handling

**Usage**:
```bash
# Start secure API
python -m uvicorn backend.app_secure:app --port 8030

# Login
TOKEN=$(curl -X POST http://localhost:8030/login \
  -d '{"username":"user"}' | jq -r '.access_token')

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8030/metrics
```

**Configuration**:
```bash
export SECRET_KEY="your-secret-key"
export TOKEN_EXPIRE_MINUTES=30
```

---

## 📊 Testing Coverage

**New Tests Added** (test_integrations.py):
- ✅ MLflow tracker creation & functionality
- ✅ Drift detection (covariate, label shift)
- ✅ JWT token creation & verification
- ✅ DVC manager initialization

**Total Test Count**: 23/23 ✅ PASSING

```bash
pytest tests/ -v
# 10 core tests + 13 integration tests = 23 total
```

---

## 🔄 Version History

- **v0.1.0** (Initial): Core MLOps + Agentic AI
- **v0.2.0** (Current): + MLflow + Drift Detection + DVC + Security
- **v0.3.0** (Planned): Kubeflow + KServe
- **v1.0.0** (Target): Complete production platform

---

## 📦 Dependencies Added

```toml
mlflow>=2.0.0
dvc>=3.0.0
dvc[s3]>=3.0.0
dvc[gs]>=3.0.0
python-jose[cryptography]>=3.3.0
pyjwt>=2.8.0
pyyaml>=6.0
```

---

## 🚀 Next Steps (v0.3.0)

1. **Kubeflow Pipelines** - Distributed training orchestration
2. **KServe Integration** - Production model serving
3. **Monitoring Dashboard** - Prometheus + Grafana
4. **IAM & RBAC** - Role-based access control
5. **Audit Logging** - Compliance tracking

---

## 📝 Configuration

### MLflow
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_TRACKING_INSECURE_TLS=true
```

### DVC
```bash
dvc remote add myremote s3://my-bucket
dvc remote default myremote
```

### Security
```bash
export SECRET_KEY="your-super-secret-key-min-32-chars"
export TOKEN_EXPIRE_MINUTES=60
```

---

## 🔗 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/)
- [DVC Docs](https://dvc.org/docs)
- [JWT Best Practices](https://tools.ietf.org/html/rfc7519)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
