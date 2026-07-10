"""Educational, local-first vision-action transformer package."""

from vat_mini.config import ExperimentConfig, TrackingConfig, load_config
from vat_mini.model import VisionActionTransformer

__all__ = ["ExperimentConfig", "TrackingConfig", "VisionActionTransformer", "load_config"]
__version__ = "0.1.0"
