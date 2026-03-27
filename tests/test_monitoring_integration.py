"""Tests for Kubeflow, KServe, and monitoring integrations."""

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from agentic_mlops.kubeflow_integration import (
    KubeflowConfig,
    PipelineStep,
    KubeflowPipelineBuilder,
    HyperparameterTuningKF,
    create_ml_pipeline
)

from agentic_mlops.kserve_integration import (
    KServeConfig,
    KServeModelServer,
    ModelRegistry
)

from agentic_mlops.prometheus_metrics import (
    MetricConfig,
    PrometheusMetrics,
    PrometheusExporter,
    track_prediction,
    track_training
)

from agentic_mlops.grafana_dashboards import (
    create_ml_monitoring_dashboard,
    create_training_dashboard,
    create_drift_detection_dashboard,
    create_alerts_config,
    create_prometheus_config
)


# ===== Kubeflow Tests =====

class TestKubeflowConfig:
    """Test Kubeflow configuration."""
    
    def test_default_config(self):
        """Test default Kubeflow config."""
        config = KubeflowConfig()
        assert config.namespace == "kubeflow"
        assert config.pipeline_name
        assert config.version == "1.0"
        assert config.host

    def test_custom_config(self):
        """Test custom Kubeflow config."""
        config = KubeflowConfig(
            namespace="custom",
            pipeline_name="test-pipeline",
            version="2.0.0"
        )
        assert config.namespace == "custom"
        assert config.pipeline_name == "test-pipeline"
        assert config.version == "2.0.0"


class TestPipelineStep:
    """Test pipeline step definition."""
    
    def test_step_creation(self):
        """Test creating a pipeline step."""
        step = PipelineStep(
            name="train",
            image="python:3.9",
            command=["python", "train.py"],
            args=["--epochs", "10"]
        )
        assert step.name == "train"
        assert step.image == "python:3.9"
        assert step.command == ["python", "train.py"]
        assert step.args == ["--epochs", "10"]

    def test_step_with_io(self):
        """Test step with inputs/outputs."""
        step = PipelineStep(
            name="evaluate",
            image="python:3.9",
            command=["python", "eval.py"],
            args=[],
            inputs={"model.pkl": "/data/model.pkl"},
            outputs={"metrics.json": "/data/metrics.json"}
        )
        assert step.inputs == {"model.pkl": "/data/model.pkl"}
        assert step.outputs == {"metrics.json": "/data/metrics.json"}


class TestKubeflowPipelineBuilder:
    """Test Kubeflow pipeline builder."""
    
    def test_builder_initialization(self):
        """Test pipeline builder init."""
        config = KubeflowConfig()
        builder = KubeflowPipelineBuilder(config)
        assert builder.config.pipeline_name == "mlops-pipeline"
        assert len(builder.steps) == 0

    def test_add_step(self):
        """Test adding steps."""
        builder = KubeflowPipelineBuilder()
        builder.add_step("step1", "python:3.9", ["python", "script.py"], [])
        
        assert len(builder.steps) == 1
        assert builder.steps[0].name == "step1"

    def test_chaining_steps(self):
        """Test method chaining."""
        builder = KubeflowPipelineBuilder()
        result = builder.add_step("s1", "img", ["cmd"], []).add_step(
            "s2", "img", ["cmd"], []
        )
        
        assert result == builder
        assert len(builder.steps) == 2

    def test_build_argo_workflow(self):
        """Test Argo workflow generation."""
        builder = KubeflowPipelineBuilder()
        builder.add_step("train", "python:3.9", ["python", "train.py"], [])
        
        workflow = builder.build_argo_workflow()
        
        assert workflow["apiVersion"] == "argoproj.io/v1alpha1"
        assert workflow["kind"] == "Workflow"
        assert workflow["metadata"]["namespace"] == "kubeflow"
        assert len(workflow["spec"]["templates"]) > 0

    def test_save_workflow_yaml(self):
        """Test saving workflow to YAML."""
        builder = KubeflowPipelineBuilder()
        builder.add_step("step1", "python:3.9", ["python"], [])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "workflow.yaml"
            builder.save_workflow(str(yaml_path))
            assert yaml_path.exists()
            
            with open(yaml_path) as f:
                content = yaml.safe_load(f)
                assert content["kind"] == "Workflow"


class TestHyperparameterTuning:
    """Test Katib hyperparameter tuning."""
    
    def test_katib_spec_creation(self):
        """Test creating Katib experiment spec."""
        tuning = HyperparameterTuningKF("test-exp")
        spec = tuning.create_trial_spec(
            model_image="python:3.9",
            parameters={"learning_rate": [0.001, 0.01, 0.1]},
            objective="Maximize",
            metric="accuracy"
        )
        
        assert spec["apiVersion"] == "kubeflow.org/v1beta1"
        assert spec["kind"] == "Experiment"
        assert spec["metadata"]["name"] == "test-exp"

    def test_trial_spec_parameters(self):
        """Test trial spec with multiple parameters."""
        tuning = HyperparameterTuningKF("multi-param")
        spec = tuning.create_trial_spec(
            model_image="python:3.9",
            parameters={
                "learning_rate": [0.001, 0.1],
                "batch_size": [16.0, 256.0]
            }
        )
        
        assert len(spec["spec"]["parameters"]) == 2


class TestCreateMLPipeline:
    """Test ML pipeline creation helper."""
    
    def test_create_pipeline(self):
        """Test creating full ML pipeline."""
        pipeline = create_ml_pipeline()
        
        assert isinstance(pipeline, KubeflowPipelineBuilder)
        assert len(pipeline.steps) >= 3
        
        step_names = [s.name for s in pipeline.steps]
        assert "prepare-data" in step_names
        assert "train-model" in step_names
        assert "evaluate-model" in step_names


# ===== KServe Tests =====

class TestKServeConfig:
    """Test KServe configuration."""
    
    def test_default_kserve_config(self):
        """Test default KServe config."""
        config = KServeConfig()
        assert config.namespace == "kserve-inference"
        assert config.model_format == "sklearn"
        assert config.storage_uri == "file:///models"

    def test_custom_kserve_config(self):
        """Test custom KServe config."""
        config = KServeConfig(
            namespace="production",
            model_format="tensorflow",
            storage_uri="s3://bucket/models"
        )
        assert config.namespace == "production"
        assert config.model_format == "tensorflow"


class TestKServeModelServer:
    """Test KServe model serving."""
    
    def test_model_server_init(self):
        """Test model server initialization."""
        server = KServeModelServer("my-model")
        assert server.model_name == "my-model"
        assert server.config.model_format == "sklearn"

    def test_create_sklearn_service(self):
        """Test creating sklearn inference service."""
        server = KServeModelServer("sklearn-model")
        service = server.create_inference_service("file:///models/model.pkl")
        
        assert service["kind"] == "InferenceService"
        assert service["metadata"]["name"] == "sklearn-model"
        assert "predictor" in service["spec"]

    def test_create_tensorflow_service(self):
        """Test creating TensorFlow inference service."""
        config = KServeConfig(model_format="tensorflow")
        server = KServeModelServer("tf-model", config)
        service = server.create_inference_service("s3://bucket/tf-model")
        
        assert "tensorflow" in service["spec"]["predictor"]

    def test_canary_deployment(self):
        """Test canary deployment setup."""
        server = KServeModelServer("canary-model")
        server.create_inference_service("file:///model.pkl")
        updated = server.add_canary_deployment(canary_traffic_percent=20)
        
        assert updated["spec"]["predictor"]["canaryTrafficPercent"] == 20

    def test_auto_scaling(self):
        """Test auto-scaling configuration."""
        server = KServeModelServer("scalable-model")
        server.create_inference_service("file:///model.pkl")
        updated = server.add_auto_scaling(min_replicas=2, max_replicas=20)
        
        assert updated["spec"]["predictor"]["minReplicas"] == 2
        assert updated["spec"]["predictor"]["maxReplicas"] == 20

    def test_prediction_url(self):
        """Test getting prediction URL."""
        server = KServeModelServer("test-model")
        url = server.get_prediction_url()
        
        assert "test-model" in url
        assert ":8080" in url
        assert "predict" in url


class TestModelRegistry:
    """Test model registry."""
    
    def test_registry_init(self):
        """Test registry initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            assert registry.registry_path == Path(tmpdir)

    def test_register_model(self):
        """Test registering a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            # Create dummy model file
            model_path = Path(tmpdir) / "model.pkl"
            model_path.write_text("model")
            
            result = registry.register_model(
                "my-model",
                "1.0.0",
                str(model_path),
                {"accuracy": 0.95}
            )
            assert result is True

    def test_get_model(self):
        """Test retrieving registered model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            model_path = Path(tmpdir) / "model.pkl"
            model_path.write_text("model")
            
            registry.register_model("test-model", "1.0.0", str(model_path), {"acc": 0.9})
            model = registry.get_model("test-model", "1.0.0")
            
            assert model is not None
            assert model["version"] == "1.0.0"

    def test_promote_model(self):
        """Test model promotion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            model_path = Path(tmpdir) / "model.pkl"
            model_path.write_text("model")
            
            registry.register_model("test-model", "1.0.0", str(model_path), {})
            registry.promote_model("test-model", "1.0.0", "production")
            
            model = registry.get_model("test-model", "1.0.0")
            assert model["status"] == "production"


# ===== Prometheus Metrics Tests =====

class TestPrometheusMetrics:
    """Test Prometheus metrics collection."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PrometheusMetrics()
        
        assert "predictions_total" in metrics.metrics
        assert "active_models" in metrics.metrics
        assert "prediction_latency_ms" in metrics.metrics

    def test_increment_counter(self):
        """Test incrementing counter."""
        metrics = PrometheusMetrics()
        metrics.increment_counter("predictions_total", 5)
        
        assert metrics.metrics["predictions_total"]["value"] == 5

    def test_set_gauge(self):
        """Test setting gauge."""
        metrics = PrometheusMetrics()
        metrics.set_gauge("active_models", 3)
        
        assert metrics.metrics["active_models"]["value"] == 3

    def test_observe_histogram(self):
        """Test histogram observation."""
        metrics = PrometheusMetrics()
        metrics.observe_histogram("prediction_latency_ms", 125)
        
        assert 125 in metrics.metrics["prediction_latency_ms"]["values"]

    def test_record_prediction(self):
        """Test recording prediction."""
        metrics = PrometheusMetrics()
        metrics.record_prediction(latency_ms=50, success=True, accuracy=0.92, model_name="test")
        
        assert metrics.metrics["predictions_total"]["value"] >= 1
        assert metrics.metrics["model_accuracy"]["value"] == 0.92

    def test_record_drift(self):
        """Test recording drift."""
        metrics = PrometheusMetrics()
        metrics.record_drift(0.75, drift_type="covariate")
        
        assert metrics.metrics["data_drift_score"]["value"] == 0.75

    def test_record_retrain(self):
        """Test recording retrain."""
        metrics = PrometheusMetrics()
        metrics.record_retrain("model1", "drift_detected")
        
        assert metrics.metrics["model_retrains_total"]["value"] >= 1

    def test_prometheus_format(self):
        """Test Prometheus text format output."""
        metrics = PrometheusMetrics()
        metrics.increment_counter("predictions_total", 10)
        
        output = metrics.get_prometheus_format()
        
        assert "predictions_total" in output
        assert "HELP" in output
        assert "TYPE" in output


class TestPrometheusDecorators:
    """Test Prometheus decorators."""
    
    def test_track_prediction_decorator(self):
        """Test prediction tracking decorator."""
        metrics = PrometheusMetrics()
        
        @track_prediction(metrics, model_name="test")
        def predict():
            return {"prediction": 0.5}
        
        result = predict()
        assert result == {"prediction": 0.5}
        assert metrics.metrics["predictions_total"]["value"] >= 1

    def test_track_training_decorator(self):
        """Test training tracking decorator."""
        metrics = PrometheusMetrics()
        
        @track_training(metrics, model_name="test")
        def train():
            return {"accuracy": 0.95}
        
        result = train()
        assert result == {"accuracy": 0.95}


class TestPrometheusExporter:
    """Test Prometheus exporter."""
    
    def test_export_text(self):
        """Test exporting metrics as text."""
        metrics = PrometheusMetrics()
        metrics.increment_counter("predictions_total", 5)
        
        exporter = PrometheusExporter(metrics)
        text = exporter.export_text()
        
        assert "predictions_total" in text

    def test_export_json(self):
        """Test exporting metrics as JSON."""
        metrics = PrometheusMetrics()
        exporter = PrometheusExporter(metrics)
        json_data = exporter.export_json()
        
        assert isinstance(json_data, dict)
        assert "predictions_total" in json_data

    def test_generate_dashboard(self):
        """Test generating Grafana dashboard."""
        metrics = PrometheusMetrics()
        exporter = PrometheusExporter(metrics)
        dashboard = exporter.generate_dashboard_json()
        
        assert "dashboard" in dashboard
        assert "panels" in dashboard["dashboard"]
        assert len(dashboard["dashboard"]["panels"]) > 0


# ===== Grafana Dashboard Tests =====

class TestGrafonaDashboards:
    """Test Grafana dashboard configuration."""
    
    def test_ml_monitoring_dashboard(self):
        """Test ML monitoring dashboard creation."""
        dashboard = create_ml_monitoring_dashboard()
        
        assert "dashboard" in dashboard
        assert dashboard["dashboard"]["title"] == "ML Model Monitoring"
        assert len(dashboard["dashboard"]["panels"]) > 0

    def test_training_dashboard(self):
        """Test training pipeline dashboard."""
        dashboard = create_training_dashboard()
        
        assert dashboard["dashboard"]["title"] == "ML Training Pipeline"
        assert "panels" in dashboard["dashboard"]

    def test_drift_detection_dashboard(self):
        """Test drift detection dashboard."""
        dashboard = create_drift_detection_dashboard()
        
        assert dashboard["dashboard"]["title"] == "Data Drift Monitoring"
        assert len(dashboard["dashboard"]["panels"]) >= 5

    def test_alerts_config(self):
        """Test alert rules configuration."""
        alerts = create_alerts_config()
        
        assert "groups" in alerts
        assert len(alerts["groups"]) > 0
        assert len(alerts["groups"][0]["rules"]) >= 5

    def test_prometheus_config(self):
        """Test Prometheus configuration."""
        config = create_prometheus_config()
        
        assert "global" in config
        assert "scrape_configs" in config
        assert "alerting" in config
        assert len(config["scrape_configs"]) > 0


# ===== Integration Tests =====

class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_kubeflow_to_kserve_pipeline(self):
        """Test pipeline from Kubeflow to KServe serving."""
        # Create pipeline
        pipeline = create_ml_pipeline()
        assert len(pipeline.steps) >= 3
        
        # Get workflow
        workflow = pipeline.build_argo_workflow()
        assert workflow["kind"] == "Workflow"
        
        # Create model server
        server = KServeModelServer("full-pipeline-model")
        service = server.create_inference_service("file:///models/model.pkl")
        assert service["kind"] == "InferenceService"

    def test_monitoring_integration(self):
        """Test monitoring integration."""
        # Setup metrics
        metrics = PrometheusMetrics()
        
        # Simulate predictions
        for _ in range(10):
            metrics.record_prediction(latency_ms=50, success=True, model_name="prod")
        
        # Export
        exporter = PrometheusExporter(metrics)
        dashboard = exporter.generate_dashboard_json()
        
        assert "dashboard" in dashboard
        assert metrics.metrics["predictions_total"]["value"] >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
