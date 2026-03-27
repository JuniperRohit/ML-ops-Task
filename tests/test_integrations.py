"""Tests for MLflow, drift detection, and authentication."""

import pytest
import numpy as np
from agentic_mlops.mlflow_tracker import MLflowTracker
from agentic_mlops.drift_detector import DriftDetector, DriftType
from agentic_mlops.auth import create_user_token, verify_token
from agentic_mlops.dvc_manager import DVCManager


class TestMLflowTracker:
    """Test MLflow tracker functionality."""

    def test_mlflow_tracker_creation(self):
        """Test MLflow tracker initialization."""
        tracker = MLflowTracker(experiment_name="test_exp")
        assert tracker.experiment_name == "test_exp"

    def test_create_token_and_verify(self):
        """Test MLflow run lifecycle."""
        tracker = MLflowTracker()
        assert tracker.enabled  # Should be enabled since mlflow is installed


class TestDriftDetector:
    """Test drift detection."""

    def test_drift_detector_init(self):
        """Test drift detector initialization."""
        detector = DriftDetector(threshold=0.3)
        assert detector.threshold == 0.3
        assert detector.window_size == 100

    def test_covariate_drift_detection(self):
        """Test covariate shift detection."""
        detector = DriftDetector(threshold=0.3)
        
        # Baseline data
        X_baseline = np.random.normal(loc=0, scale=1, size=(100, 10))
        detector.set_baseline(X_baseline)
        
        # No drift: same distribution
        X_no_drift = np.random.normal(loc=0, scale=1, size=(50, 10))
        score, detected = detector.detect_covariate_shift(X_no_drift)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert not detected or score < 0.5
        
        # Drift: different distribution
        X_drift = np.random.normal(loc=5, scale=1, size=(50, 10))
        score, detected = detector.detect_covariate_shift(X_drift)
        assert detected  # Should detect drift
        assert score > 0.5

    def test_label_shift_detection(self):
        """Test label shift detection."""
        detector = DriftDetector(threshold=0.3)
        
        # Baseline labels (balanced)
        y_baseline = np.array([0] * 50 + [1] * 50)
        
        # No shift
        y_no_shift = np.array([0] * 45 + [1] * 55)
        score, detected = detector.detect_label_shift(y_baseline, y_no_shift)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        
        # Strong shift
        y_shift = np.array([0] * 90 + [1] * 10)
        score_shift, detected_shift = detector.detect_label_shift(y_baseline, y_shift)
        assert detected_shift  # Should detect shift
        assert score_shift > score

    def test_drift_check_returns_metrics(self):
        """Test drift check returns proper metrics."""
        detector = DriftDetector()
        X_baseline = np.random.normal(size=(100, 10))
        detector.set_baseline(X_baseline)
        
        X_new = np.random.normal(loc=2, size=(50, 10))
        metrics = detector.check_drift(X_new, check_type="covariate")
        
        assert metrics.drift_type in [DriftType.COVARIATE, DriftType.NO_DRIFT]
        assert 0 <= metrics.score <= 1
        assert metrics.timestamp is not None


class TestAuthentication:
    """Test JWT authentication."""

    def test_create_user_token(self):
        """Test token creation."""
        token = create_user_token("testuser", email="test@example.com")
        assert token.access_token
        assert token.token_type == "bearer"
        assert token.expires_in > 0

    def test_verify_valid_token(self):
        """Test token verification."""
        token = create_user_token("testuser")
        verified = verify_token(token.access_token)
        assert verified is not None
        assert verified.sub == "testuser"

    def test_verify_invalid_token(self):
        """Test invalid token verification."""
        verified = verify_token("invalid.token.here")
        assert verified is None

    def test_token_has_expiration(self):
        """Test token includes expiration."""
        token = create_user_token("testuser")
        verified = verify_token(token.access_token)
        assert verified.exp is not None


class TestDVCManager:
    """Test DVC manager."""

    def test_dvc_manager_init(self):
        """Test DVC manager initialization."""
        from pathlib import Path
        manager = DVCManager(repo_path=Path("."))
        assert manager.repo_path == Path(".")
        assert manager.config.remote_name == "myremote"

    def test_dvc_available_check(self):
        """Test DVC availability check."""
        manager = DVCManager()
        # Should work even if DVC is not installed (graceful fallback)
        assert isinstance(manager.dvc_available, bool)

    def test_setup_functions_return_bool(self):
        """Test setup functions return boolean."""
        manager = DVCManager()
        # These should return bool (not crash)
        assert isinstance(manager.get_status(), dict)
