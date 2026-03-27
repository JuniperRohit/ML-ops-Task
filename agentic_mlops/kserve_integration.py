"""KServe integration for production model serving on Kubernetes."""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class KServeConfig:
    """KServe configuration."""
    namespace: str = "kserve-inference"
    model_format: str = "sklearn"  # sklearn, tensorflow, pytorch, xgboost
    storage_uri: str = "file:///models"
    api_version: str = "serving.kserve.io/v1beta1"


class KServeModelServer:
    """Manage KServe InferenceService for model serving."""
    
    def __init__(self, model_name: str, config: Optional[KServeConfig] = None):
        """
        Initialize KServe model server.
        
        Args:
            model_name: Name of model to serve
            config: KServe configuration
        """
        self.model_name = model_name
        self.config = config or KServeConfig()
        self.inference_service = None

    def create_inference_service(
        self,
        model_uri: str,
        resources_request: Optional[Dict] = None,
        resources_limit: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create KServe InferenceService specification.
        
        Args:
            model_uri: Model location (e.g., file:///models/model.pkl)
            resources_request: Resource requests (cpu, memory)
            resources_limit: Resource limits
            
        Returns:
            InferenceService spec
        """
        if resources_request is None:
            resources_request = {"cpu": "100m", "memory": "128Mi"}
        if resources_limit is None:
            resources_limit = {"cpu": "500m", "memory": "512Mi"}
        
        predictor = self._get_predictor_spec(model_uri, resources_request, resources_limit)
        
        service = {
            "apiVersion": self.config.api_version,
            "kind": "InferenceService",
            "metadata": {
                "name": self.model_name,
                "namespace": self.config.namespace
            },
            "spec": {
                "predictor": predictor
            }
        }
        
        self.inference_service = service
        return service

    def _get_predictor_spec(
        self,
        model_uri: str,
        resources_request: Dict,
        resources_limit: Dict
    ) -> Dict[str, Any]:
        """Get predictor specification based on model format."""
        
        if self.config.model_format == "sklearn":
            return {
                "sklearn": {
                    "storageUri": model_uri,
                    "resources": {
                        "requests": resources_request,
                        "limits": resources_limit
                    }
                }
            }
        
        elif self.config.model_format == "tensorflow":
            return {
                "tensorflow": {
                    "storageUri": model_uri,
                    "resources": {
                        "requests": resources_request,
                        "limits": resources_limit
                    }
                }
            }
        
        elif self.config.model_format == "pytorch":
            return {
                "pytorch": {
                    "storageUri": model_uri,
                    "modelClassName": "model.Model",
                    "resources": {
                        "requests": resources_request,
                        "limits": resources_limit
                    }
                }
            }
        
        else:
            raise ValueError(f"Unsupported model format: {self.config.model_format}")

    def add_canary_deployment(
        self,
        canary_traffic_percent: int = 10
    ) -> Dict[str, Any]:
        """
        Add canary deployment strategy.
        
        Args:
            canary_traffic_percent: Traffic % for canary model
            
        Returns:
            Updated service spec
        """
        if not self.inference_service:
            raise ValueError("Create inference service first")
        
        self.inference_service["spec"]["predictor"]["canaryTrafficPercent"] = canary_traffic_percent
        return self.inference_service

    def add_auto_scaling(
        self,
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_cpu_utilization: int = 70
    ) -> Dict[str, Any]:
        """
        Enable auto-scaling for the model.
        
        Args:
            min_replicas: Minimum replicas
            max_replicas: Maximum replicas
            target_cpu_utilization: CPU utilization target %
            
        Returns:
            Updated service spec
        """
        if not self.inference_service:
            raise ValueError("Create inference service first")
        
        self.inference_service["spec"]["predictor"]["autoscalingTarget"] = target_cpu_utilization
        self.inference_service["spec"]["predictor"]["minReplicas"] = min_replicas
        self.inference_service["spec"]["predictor"]["maxReplicas"] = max_replicas
        
        return self.inference_service

    def deploy_to_kserve(self) -> bool:
        """
        Deploy model to KServe cluster.
        
        Requires:
        - kubectl configured
        - KServe installed on cluster
        
        Returns:
            True if successful
        """
        import subprocess
        import yaml
        
        if not self.inference_service:
            raise ValueError("Create inference service first")
        
        try:
            # Save to YAML
            yaml_path = f"infer-{self.model_name}.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(self.inference_service, f)
            
            # Deploy
            result = subprocess.run(
                ["kubectl", "apply", "-f", yaml_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"Model {self.model_name} deployed to KServe")
                return True
            else:
                print(f"Deployment failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("kubectl not found. Need Kubernetes cluster with KServe.")
            return False

    def get_prediction_url(self) -> str:
        """Get the prediction endpoint URL."""
        return f"http://{self.model_name}.{self.config.namespace}.svc.cluster.local:8080/v1/models/{self.model_name}:predict"

    def get_service_config(self) -> Dict[str, str]:
        """Get service configuration."""
        return {
            "model_name": self.model_name,
            "namespace": self.config.namespace,
            "format": self.config.model_format,
            "storage_uri": self.config.storage_uri,
            "prediction_url": self.get_prediction_url()
        }


class ModelRegistry:
    """Simple model registry for versioning and management."""
    
    def __init__(self, registry_path: str = "./model_registry"):
        """Initialize model registry."""
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models_file = self.registry_path / "models.json"
        self.models = self._load_models()

    def _load_models(self) -> Dict:
        """Load models from registry."""
        if self.models_file.exists():
            with open(self.models_file) as f:
                return json.load(f)
        return {}

    def _save_models(self):
        """Save models to registry."""
        with open(self.models_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metrics: Dict[str, float],
        tags: Optional[Dict] = None
    ) -> bool:
        """
        Register a model in the registry.
        
        Args:
            model_name: Model name
            version: Model version
            model_path: Path to model file
            metrics: Model metrics (accuracy, etc.)
            tags: Model tags
            
        Returns:
            True if successful
        """
        key = f"{model_name}:{version}"
        
        self.models[key] = {
            "name": model_name,
            "version": version,
            "path": model_path,
            "metrics": metrics,
            "tags": tags or {},
            "created": str(Path(model_path).stat().st_mtime),
            "status": "registered"
        }
        
        self._save_models()
        print(f"Model {key} registered")
        return True

    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[Dict]:
        """Get model from registry."""
        if version:
            key = f"{model_name}:{version}"
            return self.models.get(key)
        
        # Return latest version
        matching = [v for k, v in self.models.items() if k.startswith(f"{model_name}:")]
        return matching[-1] if matching else None

    def promote_model(self, model_name: str, version: str, stage: str = "production") -> bool:
        """
        Promote model to production.
        
        Args:
            model_name: Model name
            version: Model version
            stage: Target stage (staging, production)
            
        Returns:
            True if successful
        """
        key = f"{model_name}:{version}"
        if key not in self.models:
            return False
        
        self.models[key]["status"] = stage
        self._save_models()
        print(f"Model {key} promoted to {stage}")
        return True

    def list_models(self, model_name: Optional[str] = None) -> Dict:
        """List models in registry."""
        if model_name:
            return {k: v for k, v in self.models.items() if k.startswith(f"{model_name}:")}
        return self.models

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_models": len(self.models),
            "models": list(self.models.keys()),
            "registry_path": str(self.registry_path)
        }
