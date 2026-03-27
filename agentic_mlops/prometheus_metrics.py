"""Prometheus metrics integration for model monitoring."""

from typing import Callable, Dict, Any, Optional
from functools import wraps
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MetricConfig:
    """Prometheus metric configuration."""
    namespace: str = "mlops"
    subsystem: str = "model"


class PrometheusMetrics:
    """Prometheus metrics collector for ML pipelines."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        """
        Initialize Prometheus metrics.
        
        Args:
            config: Metric configuration
        """
        self.config = config or MetricConfig()
        self.metrics = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize default metrics."""
        # Counter: total predictions
        self.metrics["predictions_total"] = {
            "type": "counter",
            "value": 0,
            "help": "Total number of predictions",
            "labels": {}
        }
        
        # Gauge: active models
        self.metrics["active_models"] = {
            "type": "gauge",
            "value": 0,
            "help": "Number of active models",
            "labels": {}
        }
        
        # Histogram: prediction latency (ms)
        self.metrics["prediction_latency_ms"] = {
            "type": "histogram",
            "buckets": [10, 50, 100, 500, 1000, 5000],
            "values": [],
            "help": "Prediction latency in milliseconds",
            "labels": {}
        }
        
        # Gauge: model accuracy
        self.metrics["model_accuracy"] = {
            "type": "gauge",
            "value": 0,
            "help": "Model accuracy score",
            "labels": {}
        }
        
        # Counter: prediction errors
        self.metrics["prediction_errors_total"] = {
            "type": "counter",
            "value": 0,
            "help": "Total prediction errors",
            "labels": {}
        }
        
        # Gauge: data drift
        self.metrics["data_drift_score"] = {
            "type": "gauge",
            "value": 0,
            "help": "Data drift detection score",
            "labels": {}
        }
        
        # Counter: model retrains
        self.metrics["model_retrains_total"] = {
            "type": "counter",
            "value": 0,
            "help": "Total model retrains",
            "labels": {}
        }

    def increment_counter(
        self,
        metric_name: str,
        value: float = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {
                "type": "counter",
                "value": 0,
                "labels": {}
            }
        
        self.metrics[metric_name]["value"] += value
        if labels:
            self.metrics[metric_name]["labels"].update(labels)

    def set_gauge(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {
                "type": "gauge",
                "value": 0,
                "labels": {}
            }
        
        self.metrics[metric_name]["value"] = value
        if labels:
            self.metrics[metric_name]["labels"].update(labels)

    def observe_histogram(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a histogram observation."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {
                "type": "histogram",
                "buckets": [10, 50, 100, 500, 1000, 5000],
                "values": [],
                "labels": {}
            }
        
        self.metrics[metric_name]["values"].append(value)
        if labels:
            self.metrics[metric_name]["labels"].update(labels)

    def record_prediction(
        self,
        latency_ms: float,
        success: bool = True,
        accuracy: Optional[float] = None,
        model_name: Optional[str] = None
    ):
        """Record a model prediction event."""
        self.increment_counter("predictions_total", labels={"model": model_name or "unknown"})
        self.observe_histogram("prediction_latency_ms", latency_ms, labels={"model": model_name or "unknown"})
        
        if not success:
            self.increment_counter("prediction_errors_total", labels={"model": model_name or "unknown"})
        
        if accuracy is not None:
            self.set_gauge("model_accuracy", accuracy, labels={"model": model_name or "unknown"})

    def record_drift(self, drift_score: float, drift_type: str = "covariate"):
        """Record data drift detection."""
        self.set_gauge("data_drift_score", drift_score, labels={"type": drift_type})

    def record_retrain(self, model_name: str, reason: str = "drift_detected"):
        """Record model retrain event."""
        self.increment_counter("model_retrains_total", labels={"model": model_name, "reason": reason})

    def set_active_models(self, count: int):
        """Set number of active models."""
        self.set_gauge("active_models", float(count))

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return self.metrics

    def get_metric(self, metric_name: str) -> Optional[Dict]:
        """Get a specific metric."""
        return self.metrics.get(metric_name)

    def get_prometheus_format(self) -> str:
        """Format metrics in Prometheus text format."""
        lines = []
        
        for name, metric in self.metrics.items():
            # Add HELP and TYPE
            lines.append(f"# HELP {name} {metric.get('help', '')}")
            lines.append(f"# TYPE {name} {metric.get('type')}")
            
            # Add metric values
            if metric.get("type") == "counter" or metric.get("type") == "gauge":
                label_str = self._format_labels(metric.get("labels", {}))
                lines.append(f"{name}{{{label_str}}} {metric.get('value')}")
            
            elif metric.get("type") == "histogram":
                labels = metric.get("labels", {})
                values = metric.get("values", [])
                
                if values:
                    # Calculate histogram buckets
                    buckets = metric.get("buckets", [])
                    for bucket in buckets:
                        count = sum(1 for v in values if v <= bucket)
                        label_str = self._format_labels({**labels, "le": str(bucket)})
                        lines.append(f"{name}_bucket{{{label_str}}} {count}")
                    
                    # Total count
                    label_str = self._format_labels({**labels, "le": "+Inf"})
                    lines.append(f"{name}_bucket{{{label_str}}} {len(values)}")
                    
                    # Sum
                    label_str = self._format_labels(labels)
                    lines.append(f"{name}_sum{{{label_str}}} {sum(values)}")
                    
                    # Count
                    lines.append(f"{name}_count{{{label_str}}} {len(values)}")
        
        return "\n".join(lines)

    @staticmethod
    def _format_labels(labels: Dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        items = [f'{k}="{v}"' for k, v in labels.items()]
        return ", ".join(items)


def track_prediction(metrics: PrometheusMetrics, model_name: Optional[str] = None):
    """
    Decorator to track prediction metrics.
    
    Args:
        metrics: PrometheusMetrics instance
        model_name: Model name for labeling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                metrics.record_prediction(latency_ms, success=True, model_name=model_name)
                return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                metrics.record_prediction(latency_ms, success=False, model_name=model_name)
                raise
        return wrapper
    return decorator


def track_training(metrics: PrometheusMetrics, model_name: Optional[str] = None):
    """
    Decorator to track training metrics.
    
    Args:
        metrics: PrometheusMetrics instance
        model_name: Model name for labeling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration_s = time.time() - start_time
            
            # Track training time
            metrics.observe_histogram(
                "training_duration_seconds",
                duration_s,
                labels={"model": model_name or "unknown"}
            )
            
            return result
        return wrapper
    return decorator


class PrometheusExporter:
    """Export metrics for Prometheus scraping."""
    
    def __init__(self, metrics: PrometheusMetrics):
        """Initialize exporter."""
        self.metrics = metrics

    def export_text(self) -> str:
        """Export metrics in Prometheus text format."""
        return self.metrics.get_prometheus_format()

    def export_json(self) -> Dict:
        """Export metrics as JSON."""
        return self.metrics.get_metrics()

    def generate_dashboard_json(self, title: str = "ML Metrics") -> Dict:
        """Generate Grafana dashboard JSON."""
        return {
            "dashboard": {
                "title": title,
                "uid": "mlops-dashboard",
                "version": 1,
                "timezone": "browser",
                "panels": [
                    {
                        "title": "Predictions per Second",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(predictions_total[5m])",
                                "legendFormat": "predictions/sec"
                            }
                        ]
                    },
                    {
                        "title": "Prediction Latency (99th percentile)",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.99, prediction_latency_ms)",
                                "legendFormat": "p99 latency (ms)"
                            }
                        ]
                    },
                    {
                        "title": "Data Drift Score",
                        "type": "gauge",
                        "targets": [
                            {
                                "expr": "data_drift_score",
                                "legendFormat": "drift score"
                            }
                        ]
                    },
                    {
                        "title": "Model Accuracy",
                        "type": "gauge",
                        "targets": [
                            {
                                "expr": "model_accuracy",
                                "legendFormat": "accuracy"
                            }
                        ]
                    },
                    {
                        "title": "Prediction Errors",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(prediction_errors_total[5m])",
                                "legendFormat": "errors/sec"
                            }
                        ]
                    },
                    {
                        "title": "Model Retrains",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "increase(model_retrains_total[1h])",
                                "legendFormat": "retrains/hour"
                            }
                        ]
                    }
                ]
            }
        }
