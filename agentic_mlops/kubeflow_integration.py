"""Kubeflow Pipelines integration for distributed ML workflows."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class KubeflowConfig:
    """Kubeflow configuration."""
    namespace: str = "kubeflow"
    pipeline_name: str = "mlops-pipeline"
    version: str = "1.0"
    host: str = "http://localhost:3000"


@dataclass
class PipelineStep:
    """Definition of a pipeline step/task."""
    name: str
    image: str
    command: List[str]
    args: List[str]
    inputs: Optional[Dict[str, str]] = None
    outputs: Optional[Dict[str, str]] = None


class KubeflowPipelineBuilder:
    """Build and manage Kubeflow ML pipelines."""
    
    def __init__(self, config: Optional[KubeflowConfig] = None):
        """
        Initialize Kubeflow pipeline builder.
        
        Args:
            config: Kubeflow configuration
        """
        self.config = config or KubeflowConfig()
        self.steps: List[PipelineStep] = []
        self.pipeline_yaml = None

    def add_step(
        self,
        name: str,
        image: str,
        command: List[str],
        args: List[str],
        inputs: Optional[Dict[str, str]] = None,
        outputs: Optional[Dict[str, str]] = None
    ) -> "KubeflowPipelineBuilder":
        """
        Add a step to the pipeline.
        
        Args:
            name: Step name
            image: Docker image to run
            command: Command to execute
            args: Arguments to command
            inputs: Input artifacts/parameters
            outputs: Output artifacts/parameters
            
        Returns:
            Self for chaining
        """
        step = PipelineStep(
            name=name,
            image=image,
            command=command,
            args=args,
            inputs=inputs,
            outputs=outputs
        )
        self.steps.append(step)
        return self

    def build_argo_workflow(self) -> Dict[str, Any]:
        """
        Build Argo Workflow compatible with Kubeflow.
        
        Returns:
            Workflow YAML as dict
        """
        tasks = []
        
        # Build sequential pipeline
        for i, step in enumerate(self.steps):
            task = {
                "name": step.name,
                "container": {
                    "image": step.image,
                    "command": step.command,
                    "args": step.args,
                    "workingDir": "/ml"
                }
            }
            
            # Add dependencies
            if i > 0:
                task["dependencies"] = self.steps[i-1].name
            
            tasks.append(task)
        
        workflow = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "metadata": {
                "generateName": f"{self.config.pipeline_name}-",
                "namespace": self.config.namespace
            },
            "spec": {
                "entrypoint": "pipeline",
                "templates": [
                    {
                        "name": "pipeline",
                        "dag": {
                            "tasks": tasks
                        }
                    }
                ] + self._container_templates()
            }
        }
        
        self.pipeline_yaml = workflow
        return workflow

    def _container_templates(self) -> List[Dict]:
        """Generate container templates for each step."""
        templates = []
        for step in self.steps:
            templates.append({
                "name": step.name,
                "container": {
                    "image": step.image,
                    "command": step.command,
                    "args": step.args
                }
            })
        return templates

    def save_workflow(self, filepath: str = "kubeflow-pipeline.yaml"):
        """
        Save pipeline to YAML file.
        
        Args:
            filepath: Output file path
        """
        if not self.pipeline_yaml:
            self.build_argo_workflow()
        
        with open(filepath, "w") as f:
            yaml.dump(self.pipeline_yaml, f, default_flow_style=False)
        
        print(f"Pipeline saved to {filepath}")

    def deploy_to_kubeflow(self) -> bool:
        """
        Deploy pipeline to Kubeflow.
        
        Requires:
        - kubectl configured
        - Kubeflow installed on cluster
        
        Returns:
            True if successful
        """
        import subprocess
        
        try:
            # Create workflow YAML
            self.save_workflow()
            
            # Apply to cluster
            result = subprocess.run(
                ["kubectl", "apply", "-f", "kubeflow-pipeline.yaml"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("Pipeline deployed successfully")
                return True
            else:
                print(f"Deployment failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("kubectl not found. Install kubectl or run locally first.")
            return False

    def get_pipeline_config(self) -> Dict[str, str]:
        """Get pipeline configuration."""
        return {
            "name": self.config.pipeline_name,
            "namespace": self.config.namespace,
            "version": self.config.version,
            "steps": len(self.steps),
            "host": self.config.host
        }


class HyperparameterTuningKF:
    """Hyperparameter tuning using Katib (Kubeflow's tuning framework)."""
    
    def __init__(self, experiment_name: str = "hp-tuning"):
        """Initialize hyperparameter tuning."""
        self.experiment_name = experiment_name

    def create_trial_spec(
        self,
        model_image: str,
        parameters: Dict[str, List[float]],
        objective: str = "Maximize",
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Create Katib trial specification.
        
        Args:
            model_image: Docker image for training
            parameters: Parameters to tune (name -> values)
            objective: Optimize direction (Maximize/Minimize)
            metric: Metric to optimize
            
        Returns:
            Trial specification
        """
        parameters_spec = []
        for param_name, param_range in parameters.items():
            parameters_spec.append({
                "name": param_name,
                "parameterType": "double",
                "feasibleSpace": {
                    "min": str(min(param_range)),
                    "max": str(max(param_range))
                }
            })
        
        spec = {
            "apiVersion": "kubeflow.org/v1beta1",
            "kind": "Experiment",
            "metadata": {
                "name": self.experiment_name
            },
            "spec": {
                "algorithm": {
                    "algorithmName": "random"
                },
                "parallelTrialCount": 3,
                "maxTrialCount": 12,
                "maxFailedTrialCount": 3,
                "objective": {
                    "type": objective.lower(),
                    "goal": 0.99,
                    "objectiveMetricName": metric
                },
                "parameters": parameters_spec,
                "trialTemplate": {
                    "goTemplate": {
                        "rawTemplate": "yaml_template_content"
                    }
                }
            }
        }
        
        return spec


def create_ml_pipeline() -> KubeflowPipelineBuilder:
    """Create example ML pipeline."""
    builder = KubeflowPipelineBuilder()
    
    builder.add_step(
        name="prepare-data",
        image="python:3.12",
        command=["python"],
        args=["-m", "agentic_mlops.skills", "generate_dataset"],
        outputs={"data": "/ml/data/generated_data.npz"}
    )
    
    builder.add_step(
        name="train-model",
        image="python:3.12",
        command=["python"],
        args=["-m", "agentic_mlops.skills", "train_model"],
        inputs={"data": "/ml/data/generated_data.npz"},
        outputs={"model": "/ml/models/best_model.pkl"}
    )
    
    builder.add_step(
        name="evaluate-model",
        image="python:3.12",
        command=["python"],
        args=["-m", "agentic_mlops.skills", "evaluate_model"],
        inputs={"model": "/ml/models/best_model.pkl"},
        outputs={"metrics": "/ml/metrics/eval.json"}
    )
    
    return builder
