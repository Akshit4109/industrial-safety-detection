"""Central configuration management module."""
from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigManager:
    """Loads and parses project YAML configurations."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def project(self) -> Dict[str, Any]:
        return self._config.get("project", {})

    @property
    def dataset(self) -> Dict[str, Any]:
        return self._config.get("dataset", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self._config.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self._config.get("training", {})

    @property
    def inference(self) -> Dict[str, Any]:
        return self._config.get("inference", {})

    def get_all(self) -> Dict[str, Any]:
        return self._config
