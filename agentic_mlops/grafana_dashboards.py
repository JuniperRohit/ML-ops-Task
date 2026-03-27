"""Grafana dashboard templates for MLOps monitoring."""

import json
from typing import Dict, Any, List


def create_ml_monitoring_dashboard() -> Dict[str, Any]:
    """Create comprehensive ML monitoring dashboard."""
    return {
        "dashboard": {
            "title": "ML Model Monitoring",
            "description": "Real-time monitoring of ML models in production",
            "uid": "ml-monitoring",
            "version": 0,
            "timezone": "browser",
            "panels": [
                # Row 1: Key Metrics
                {
                    "id": 1,
                    "title": "Predictions (per minute)",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "rate(predictions_total[1m])",
                            "legendFormat": "{{model}}"
                        }
                    ],
                    "options": {
                        "colorMode": "value",
                        "graphMode": "area",
                        "unit": "reqps"
                    }
                },
                {
                    "id": 2,
                    "title": "Prediction Errors",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                    "targets": [
                        {
                            "expr": "rate(prediction_errors_total[1m])",
                            "legendFormat": "{{model}}"
                        }
                    ],
                    "options": {
                        "colorMode": "value",
                        "unit": "short",
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "red", "value": 10}
                            ]
                        }
                    }
                },
                {
                    "id": 3,
                    "title": "Data Drift Score",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": "data_drift_score",
                            "legendFormat": "{{type}}"
                        }
                    ],
                    "options": {
                        "orientation": "auto",
                        "showThresholdLabels": False,
                        "showThresholdMarkers": True,
                        "unit": "short",
                        "max": 1,
                        "min": 0
                    }
                },
                {
                    "id": 4,
                    "title": "Active Models",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                    "targets": [
                        {
                            "expr": "active_models",
                        }
                    ],
                    "options": {
                        "colorMode": "value",
                        "graphMode": "none"
                    }
                },
                
                # Row 2: Latency
                {
                    "id": 5,
                    "title": "Prediction Latency (p50, p95, p99)",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.50, prediction_latency_ms)",
                            "legendFormat": "p50"
                        },
                        {
                            "expr": "histogram_quantile(0.95, prediction_latency_ms)",
                            "legendFormat": "p95"
                        },
                        {
                            "expr": "histogram_quantile(0.99, prediction_latency_ms)",
                            "legendFormat": "p99"
                        }
                    ],
                    "options": {
                        "legend": {"showLegend": True, "placement": "bottom"},
                    }
                },
                
                # Row 3: Accuracy & Retrains
                {
                    "id": 6,
                    "title": "Model Accuracy",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": "model_accuracy",
                            "legendFormat": "{{model}}"
                        }
                    ],
                    "options": {
                        "legend": {"showLegend": True}
                    }
                },
                {
                    "id": 7,
                    "title": "Model Retrains",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                    "targets": [
                        {
                            "expr": "increase(model_retrains_total[1h])",
                            "legendFormat": "{{model}}"
                        }
                    ],
                    "options": {
                        "legend": {"showLegend": True}
                    }
                },
                
                # Row 4: System Health
                {
                    "id": 8,
                    "title": "Error Rate",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
                    "targets": [
                        {
                            "expr": "rate(prediction_errors_total[5m]) / rate(predictions_total[5m])",
                            "legendFormat": "{{model}}"
                        }
                    ],
                    "options": {
                        "legend": {"showLegend": True}
                    }
                },
                {
                    "id": 9,
                    "title": "Throughput (requests/sec)",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
                    "targets": [
                        {
                            "expr": "rate(predictions_total[5m])",
                            "legendFormat": "{{model}}"
                        }
                    ],
                    "options": {
                        "legend": {"showLegend": True}
                    }
                }
            ],
            "refresh": "30s",
            "time": {
                "from": "now-6h",
                "to": "now"
            }
        }
    }


def create_training_dashboard() -> Dict[str, Any]:
    """Create ML training pipeline dashboard."""
    return {
        "dashboard": {
            "title": "ML Training Pipeline",
            "description": "Monitor training jobs and model development",
            "uid": "ml-training",
            "version": 0,
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Training Job Status",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "training_job_status",
                            "format": "table"
                        }
                    ]
                },
                {
                    "id": 2,
                    "title": "Training Duration",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "training_duration_seconds",
                            "legendFormat": "{{model}}"
                        }
                    ]
                },
                {
                    "id": 3,
                    "title": "Data Quality Score",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": "data_quality_score",
                        }
                    ],
                    "options": {
                        "max": 100,
                        "min": 0,
                        "unit": "percent"
                    }
                }
            ],
            "refresh": "1m",
            "time": {
                "from": "now-24h",
                "to": "now"
            }
        }
    }


def create_drift_detection_dashboard() -> Dict[str, Any]:
    """Create data drift monitoring dashboard."""
    return {
        "dashboard": {
            "title": "Data Drift Monitoring",
            "description": "Detect and monitor data distribution shifts",
            "uid": "data-drift",
            "version": 0,
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Covariate Drift Score",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "data_drift_score{type='covariate'}"
                        }
                    ],
                    "options": {
                        "max": 1,
                        "min": 0,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 0.5},
                                {"color": "red", "value": 0.8}
                            ]
                        }
                    }
                },
                {
                    "id": 2,
                    "title": "Label Drift Score",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0},
                    "targets": [
                        {
                            "expr": "data_drift_score{type='label'}"
                        }
                    ],
                    "options": {
                        "max": 1,
                        "min": 0,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 0.5},
                                {"color": "red", "value": 0.8}
                            ]
                        }
                    }
                },
                {
                    "id": 3,
                    "title": "Concept Drift Score",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0},
                    "targets": [
                        {
                            "expr": "data_drift_score{type='concept'}"
                        }
                    ],
                    "options": {
                        "max": 1,
                        "min": 0,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 0.5},
                                {"color": "red", "value": 0.8}
                            ]
                        }
                    }
                },
                {
                    "id": 4,
                    "title": "Drift Detection Timeline",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "data_drift_score",
                            "legendFormat": "{{type}}"
                        }
                    ],
                    "options": {
                        "legend": {"showLegend": True, "placement": "bottom"}
                    }
                },
                {
                    "id": 5,
                    "title": "Features with Drift",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": "drift_feature_scores",
                            "format": "table"
                        }
                    ]
                }
            ],
            "refresh": "1m",
            "time": {
                "from": "now-7d",
                "to": "now"
            }
        }
    }


def create_alerts_config() -> Dict[str, Any]:
    """Create Prometheus alert rules."""
    return {
        "groups": [
            {
                "name": "mlops_alerts",
                "interval": "30s",
                "rules": [
                    {
                        "alert": "HighPredictionErrorRate",
                        "expr": "rate(prediction_errors_total[5m]) > 0.05",
                        "for": "5m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "High prediction error rate",
                            "description": "Error rate is {{ $value | humanizePercentage }} on {{ $labels.model }}"
                        }
                    },
                    {
                        "alert": "DataDriftDetected",
                        "expr": "data_drift_score > 0.7",
                        "for": "10m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Data drift detected",
                            "description": "Drift score {{ $value }} for {{ $labels.type }} on model {{ $labels.model }}"
                        }
                    },
                    {
                        "alert": "HighLatency",
                        "expr": "histogram_quantile(0.99, prediction_latency_ms) > 1000",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High prediction latency",
                            "description": "P99 latency is {{ $value }}ms for {{ $labels.model }}"
                        }
                    },
                    {
                        "alert": "ModelAccuracyDegraded",
                        "expr": "model_accuracy < 0.8",
                        "for": "15m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Model accuracy degraded",
                            "description": "Model {{ $labels.model }} accuracy dropped to {{ $value }}"
                        }
                    },
                    {
                        "alert": "InferenceServiceDown",
                        "expr": "up{job='kserve'} == 0",
                        "for": "5m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Inference service down",
                            "description": "KServe inference service {{ $labels.instance }} is down"
                        }
                    }
                ]
            }
        ]
    }


def create_prometheus_config() -> Dict[str, Any]:
    """Create Prometheus configuration."""
    return {
        "global": {
            "scrape_interval": "15s",
            "evaluation_interval": "15s",
            "external_labels": {
                "monitor": "mlops-monitor"
            }
        },
        "scrape_configs": [
            {
                "job_name": "prometheus",
                "static_configs": [
                    {"targets": ["localhost:9090"]}
                ]
            },
            {
                "job_name": "mlops-metrics",
                "static_configs": [
                    {"targets": ["localhost:8000/metrics"]}
                ]
            },
            {
                "job_name": "kserve",
                "static_configs": [
                    {"targets": ["kserve-inference:8080"]}
                ]
            }
        ],
        "alerting": {
            "alertmanagers": [
                {"static_configs": [{"targets": ["localhost:9093"]}]}
            ]
        },
        "rule_files": [
            "alerts.yml"
        ]
    }
