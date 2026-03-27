import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .config import DATA_DIR, MODELS_DIR
from .schemas import TaskResult, TaskStatus, TaskType


class SkillRegistry:
    _skills = {}

    @classmethod
    def register(cls, name):
        def wrapper(func):
            cls._skills[name] = func
            return func

        return wrapper

    @classmethod
    def get(cls, name):
        return cls._skills.get(name)


@SkillRegistry.register("generate_dataset")
def generate_dataset(objective: str, config: Dict[str, Any]) -> TaskResult:
    n_samples = int(config.get("n_samples", 1000))
    n_features = int(config.get("n_features", 20))
    class_sep = float(config.get("class_sep", 1.0))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative= int(max(2, n_features // 3)),
        n_redundant=int(max(0, n_features // 5)),
        n_classes=2,
        class_sep=class_sep,
        random_state=42,
    )

    data_file = DATA_DIR / "generated_data.npz"
    np.savez_compressed(data_file, X=X, y=y)

    details = {
        "sample": X.tolist()[:2],
        "target": y.tolist()[:2],
        "data_file": str(data_file),
        "shape": X.shape,
    }

    return TaskResult(name="generate_dataset", task_type=TaskType.data, status=TaskStatus.success, details=details)


@SkillRegistry.register("train_model")
def train_model(objective: str, config: Dict[str, Any]) -> TaskResult:
    data_file = config.get("data_file", str(DATA_DIR / "generated_data.npz"))
    model_file = config.get("model_file", str(MODELS_DIR / "best_model.pkl"))
    penalty = config.get("penalty", "l2")
    C = float(config.get("C", 1.0))

    npz = np.load(data_file)
    X, y = npz["X"], npz["y"]

    model = LogisticRegression(max_iter=1000, penalty=penalty, C=C, random_state=42)
    model.fit(X, y)

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    details = {
        "model_file": model_file,
        "coeff_shape": model.coef_.shape,
        "intercept": model.intercept_.tolist(),
    }

    return TaskResult(name="train_model", task_type=TaskType.training, status=TaskStatus.success, details=details)


@SkillRegistry.register("evaluate_model")
def evaluate_model(objective: str, config: Dict[str, Any]) -> TaskResult:
    data_file = config.get("data_file", str(DATA_DIR / "generated_data.npz"))
    model_file = config.get("model_file", str(MODELS_DIR / "best_model.pkl"))

    npz = np.load(data_file)
    X, y = npz["X"], npz["y"]

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    pred = model.predict(X)
    acc = float(accuracy_score(y, pred))

    details = {
        "accuracy": acc,
        "n_samples": int(X.shape[0]),
    }

    status = TaskStatus.success if acc > 0.70 else TaskStatus.failed
    return TaskResult(name="evaluate_model", task_type=TaskType.evaluation, status=status, details=details)


@SkillRegistry.register("deploy_model")
def deploy_model(objective: str, config: Dict[str, Any]) -> TaskResult:
    model_file = config.get("model_file", str(MODELS_DIR / "best_model.pkl"))
    deployment_file = Path(config.get("deployment_file", MODELS_DIR / "deployed_model.json"))

    with open(model_file, "rb") as f:
        model_bytes = f.read()

    release_data = {
        "objective": objective,
        "status": "deployed",
        "model_file": model_file,
        "artifact_bytes": len(model_bytes),
    }

    with open(deployment_file, "w", encoding="utf-8") as f:
        json.dump(release_data, f, indent=2)

    details = {"deployment_file": str(deployment_file), "artifact_bytes": len(model_bytes)}
    return TaskResult(name="deploy_model", task_type=TaskType.deployment, status=TaskStatus.success, details=details)
