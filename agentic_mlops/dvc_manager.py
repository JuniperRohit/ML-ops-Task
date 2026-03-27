"""DVC (Data Version Control) integration for data versioning."""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict


@dataclass
class DvcConfig:
    """DVC configuration."""
    remote_name: str = "myremote"
    remote_url: Optional[str] = None
    autostage: bool = True


class DVCManager:
    """Manage DVC for data versioning and pipeline tracking."""
    
    def __init__(self, repo_path: Path = Path("."), config: Optional[DvcConfig] = None):
        """
        Initialize DVC manager.
        
        Args:
            repo_path: Path to git/DVC repository
            config: DVC configuration
        """
        self.repo_path = Path(repo_path)
        self.dvc_dir = self.repo_path / ".dvc"
        self.config = config or DvcConfig()
        self.dvc_available = self._check_dvc_available()

    def _check_dvc_available(self) -> bool:
        """Check if DVC is installed."""
        try:
            result = subprocess.run(["dvc", "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def init(self) -> bool:
        """
        Initialize DVC in repository.
        
        Returns:
            True if successful
        """
        if not self.dvc_available:
            print("DVC not installed. Install with: pip install dvc")
            return False
        
        if self.dvc_dir.exists():
            print("DVC already initialized")
            return True
        
        try:
            result = subprocess.run(
                ["dvc", "init"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error initializing DVC: {e}")
            return False

    def add_data(self, filepath: str) -> bool:
        """
        Add data file to DVC tracking.
        
        Args:
            filepath: Path to data file
            
        Returns:
            True if successful
        """
        if not self.dvc_available:
            return False
        
        try:
            result = subprocess.run(
                ["dvc", "add", filepath],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error adding data to DVC: {e}")
            return False

    def configure_remote(self, remote_url: Optional[str] = None) -> bool:
        """
        Configure DVC remote storage.
        
        Args:
            remote_url: Remote storage URL (S3, GCS, etc.)
                        Format: 's3://bucket/path' or 'gs://bucket/path'
        
        Returns:
            True if successful
        """
        if not self.dvc_available:
            return False
        
        url = remote_url or self.config.remote_url
        if not url:
            print("No remote URL provided")
            return False
        
        try:
            # Add remote
            result = subprocess.run(
                ["dvc", "remote", "add", "-d", self.config.remote_name, url],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Remote might already exist, try to modify
                subprocess.run(
                    ["dvc", "remote", "modify", self.config.remote_name, "url", url],
                    cwd=self.repo_path,
                    capture_output=True
                )
            
            return True
        except Exception as e:
            print(f"Error configuring remote: {e}")
            return False

    def push_data(self, filepath: Optional[str] = None) -> bool:
        """
        Push data to remote storage.
        
        Args:
            filepath: Specific file to push (optional)
            
        Returns:
            True if successful
        """
        if not self.dvc_available:
            return False
        
        try:
            cmd = ["dvc", "push"]
            if filepath:
                cmd.append(filepath)
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error pushing data: {e}")
            return False

    def pull_data(self, filepath: Optional[str] = None) -> bool:
        """
        Pull data from remote storage.
        
        Args:
            filepath: Specific file to pull (optional)
            
        Returns:
            True if successful
        """
        if not self.dvc_available:
            return False
        
        try:
            cmd = ["dvc", "pull"]
            if filepath:
                cmd.append(filepath)
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error pulling data: {e}")
            return False

    def create_pipeline(self, pipeline_file: str = "dvc.yaml") -> bool:
        """
        Create DVC pipeline file.
        
        Pipeline format:
        ```yaml
        stages:
          data_prep:
            cmd: python prepare.py
            deps:
              - raw_data.csv
            outs:
              - processed_data.csv
          train:
            cmd: python train.py
            deps:
              - processed_data.csv
            outs:
              - model.pkl
        ```
        
        Args:
            pipeline_file: Path to DVC pipeline file
            
        Returns:
            True if file created
        """
        pipeline_config = {
            "stages": {
                "generate_data": {
                    "cmd": "python -m agentic_mlops.skills generate_dataset",
                    "outs": ["data/generated_data.npz"]
                },
                "train_model": {
                    "cmd": "python -m agentic_mlops.skills train_model",
                    "deps": ["data/generated_data.npz"],
                    "outs": ["models/best_model.pkl"]
                },
                "evaluate_model": {
                    "cmd": "python -m agentic_mlops.skills evaluate_model",
                    "deps": ["data/generated_data.npz", "models/best_model.pkl"]
                }
            }
        }
        
        try:
            with open(self.repo_path / pipeline_file, "w") as f:
                import yaml
                yaml.dump(pipeline_config, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            return False

    def repro(self) -> bool:
        """
        Run DVC pipeline (reproduce results).
        
        Returns:
            True if successful
        """
        if not self.dvc_available:
            return False
        
        try:
            result = subprocess.run(
                ["dvc", "repro"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error running DVC pipeline: {e}")
            return False

    def get_status(self) -> Dict:
        """
        Get DVC status.
        
        Returns:
            Status information
        """
        if not self.dvc_available:
            return {"status": "DVC not available"}
        
        try:
            result = subprocess.run(
                ["dvc", "status"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return {
                "status": "success" if result.returncode == 0 else "error",
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


def setup_dvc_s3(bucket_name: str, region: str = "us-east-1") -> bool:
    """
    Setup DVC with S3 remote.
    
    Args:
        bucket_name: S3 bucket name
        region: AWS region
        
    Returns:
        True if successful
    """
    manager = DVCManager()
    if not manager.dvc_available:
        print("Installing DVC with S3 support...")
        subprocess.run(["pip", "install", "dvc[s3]"], capture_output=True)
        manager = DVCManager()
    
    remote_url = f"s3://{bucket_name}"
    return manager.init() and manager.configure_remote(remote_url)


def setup_dvc_gcs(bucket_name: str) -> bool:
    """
    Setup DVC with Google Cloud Storage.
    
    Args:
        bucket_name: GCS bucket name
        
    Returns:
        True if successful
    """
    manager = DVCManager()
    if not manager.dvc_available:
        print("Installing DVC with GCS support...")
        subprocess.run(["pip", "install", "dvc[gs]"], capture_output=True)
        manager = DVCManager()
    
    remote_url = f"gs://{bucket_name}"
    return manager.init() and manager.configure_remote(remote_url)
