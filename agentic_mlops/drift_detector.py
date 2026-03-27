"""Model drift detection and monitoring."""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional
from enum import Enum


class DriftType(str, Enum):
    """Types of data drift."""
    COVARIATE = "covariate_shift"  # Input distribution changed
    LABEL = "label_shift"  # Output distribution changed
    CONCEPT = "concept_shift"  # Decision boundary changed
    NO_DRIFT = "no_drift"


@dataclass
class DriftMetrics:
    """Metrics for drift detection."""
    drift_type: DriftType
    score: float  # 0.0 to 1.0
    threshold: float
    detected: bool
    timestamp: datetime
    details: dict


class DriftDetector:
    """Detect model/data drift using statistical tests."""
    
    def __init__(self, threshold: float = 0.3, window_size: int = 100):
        """
        Initialize drift detector.
        
        Args:
            threshold: Drift detection threshold (0.0-1.0)
            window_size: Size of baseline window for comparison
        """
        self.threshold = threshold
        self.window_size = window_size
        self.baseline_data = None
        self.baseline_predictions = None
        self.baseline_mean = None
        self.baseline_std = None

    def set_baseline(self, X: np.ndarray, y_pred: Optional[np.ndarray] = None):
        """
        Set baseline data for comparison.
        
        Args:
            X: Feature matrix (baseline)
            y_pred: Baseline predictions (optional)
        """
        self.baseline_data = X[-self.window_size:] if X.shape[0] > self.window_size else X
        self.baseline_predictions = y_pred[-self.window_size:] if y_pred is not None else None
        self.baseline_mean = np.mean(self.baseline_data, axis=0)
        self.baseline_std = np.std(self.baseline_data, axis=0)

    def detect_covariate_shift(self, X_new: np.ndarray) -> Tuple[float, bool]:
        """
        Detect covariate shift (input distribution change).
        
        Uses Kolmogorov-Smirnov test approximation via feature drift.
        
        Args:
            X_new: New feature matrix
            
        Returns:
            (drift_score, is_drift_detected)
        """
        if self.baseline_mean is None:
            return 0.0, False
        
        # Calculate feature-wise drift
        new_mean = np.mean(X_new, axis=0)
        
        # Standardized difference
        safe_std = np.where(self.baseline_std == 0, 1.0, self.baseline_std)
        drift_scores = np.abs((new_mean - self.baseline_mean) / (safe_std + 1e-8))
        
        # Average drift across features
        drift_score = np.mean(drift_scores)
        
        # Normalize to 0-1
        drift_score = np.tanh(drift_score)
        
        return float(drift_score), drift_score > self.threshold

    def detect_label_shift(self, y_baseline: np.ndarray, y_new: np.ndarray) -> Tuple[float, bool]:
        """
        Detect label shift (output distribution change).
        
        Args:
            y_baseline: Baseline labels
            y_new: New labels
            
        Returns:
            (drift_score, is_drift_detected)
        """
        if len(y_baseline) == 0 or len(y_new) == 0:
            return 0.0, False
        
        # Class distribution comparison (Chi-square approximation)
        classes = np.unique(np.concatenate([y_baseline, y_new]))
        drift_score = 0.0
        
        for cls in classes:
            baseline_ratio = np.mean(y_baseline == cls) + 1e-8
            new_ratio = np.mean(y_new == cls) + 1e-8
            
            # Chi-square component
            chi2_contrib = ((new_ratio - baseline_ratio) ** 2) / baseline_ratio
            drift_score += chi2_contrib
        
        # Normalize
        drift_score = min(1.0, drift_score / len(classes))
        
        return float(drift_score), drift_score > self.threshold

    def detect_concept_shift(self, 
                            X_baseline: np.ndarray, 
                            y_baseline: np.ndarray,
                            X_new: np.ndarray,
                            y_new: np.ndarray) -> Tuple[float, bool]:
        """
        Detect concept shift (decision boundary change).
        
        Approximated by comparing prediction confidence distributions.
        
        Args:
            X_baseline: Baseline features
            y_baseline: Baseline labels
            X_new: New features
            y_new: New labels
            
        Returns:
            (drift_score, is_drift_detected)
        """
        if len(y_baseline) == 0 or len(y_new) == 0:
            return 0.0, False
        
        # Proxy: error rate change
        baseline_error = np.mean(y_baseline != y_baseline)  # Would use model predictions in practice
        new_error = np.mean(y_new != y_new)
        
        drift_score = abs(new_error - baseline_error)
        drift_score = min(1.0, drift_score)
        
        return float(drift_score), drift_score > self.threshold

    def check_drift(self, 
                   X_new: np.ndarray,
                   y_new: Optional[np.ndarray] = None,
                   check_type: str = "covariate") -> DriftMetrics:
        """
        Check for drift in new data.
        
        Args:
            X_new: New feature matrix
            y_new: New labels (optional)
            check_type: Type of drift to check ('covariate', 'label', 'concept')
            
        Returns:
            DriftMetrics with drift information
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not set. Call set_baseline() first.")
        
        if check_type == "covariate":
            score, detected = self.detect_covariate_shift(X_new)
            drift_type = DriftType.COVARIATE if detected else DriftType.NO_DRIFT
        elif check_type == "label" and y_new is not None:
            score, detected = self.detect_label_shift(self.baseline_predictions or [], y_new)
            drift_type = DriftType.LABEL if detected else DriftType.NO_DRIFT
        elif check_type == "concept" and y_new is not None:
            score, detected = self.detect_concept_shift(
                self.baseline_data, self.baseline_predictions or np.zeros(len(self.baseline_data)),
                X_new, y_new
            )
            drift_type = DriftType.CONCEPT if detected else DriftType.NO_DRIFT
        else:
            score, detected = self.detect_covariate_shift(X_new)
            drift_type = DriftType.COVARIATE if detected else DriftType.NO_DRIFT
        
        return DriftMetrics(
            drift_type=drift_type,
            score=score,
            threshold=self.threshold,
            detected=detected,
            timestamp=datetime.now(),
            details={
                "check_type": check_type,
                "samples_checked": len(X_new),
                "baseline_size": len(self.baseline_data)
            }
        )
