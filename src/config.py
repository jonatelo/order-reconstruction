"""
Configuration file for the order reconstruction pipeline.

Define the algorithm to use and pipeline parameters here.
"""

from pathlib import Path
from typing import Any, Dict

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATASET PARAMETERS
# ============================================================================
DATASET_CONFIG = {
    "sampling_rate": 93750,  # Hz
    "gear_ratio": 5.095238095,
    "nominal_turbine_speed": 536.27,  # Hz
    "bearing_factors": {
        "cage": 0.43,
        "ball": 7.05,
        "inner_race": 10.78,
        "outer_race": 8.22
    },
    "fault_band_centers": [231, 3781, 5781, 4408],  # Hz
}


# ============================================================================
# ALGORITHM SELECTION
# ============================================================================
# Import your algorithm classes here as you create them
# from src.algorithms.signal_processing import SignalProcessingAlgorithm
# from src.algorithms.statistical import StatisticalAlgorithm
# from src.algorithms.ml_based import MLBasedAlgorithm

# Define which algorithm to use
ALGORITHM_NAME = "signal_processing"  # Change this to switch algorithms

# Algorithm-specific configuration
ALGORITHM_CONFIG: Dict[str, Any] = {
    "signal_processing": {
        "features": ["rms", "kurtosis", "peak_frequency", "fault_band_power"],
        "frequency_bands": DATASET_CONFIG["fault_band_centers"],
        "window_size": None,  # Use full signal
    },
    "statistical": {
        "features": ["trend", "variance", "entropy"],
        "smoothing_window": 5,
    },
    "ml_based": {
        "model_type": "autoencoder",
        "embedding_dim": 16,
        "pretrained_model": None,
    }
}


# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
PREPROCESSING_CONFIG = {
    "remove_outliers": False,
    "normalize": True,
    "detrend": False,
    "filter_type": None,  # Options: 'lowpass', 'highpass', 'bandpass', None
    "filter_params": {},
}


# ============================================================================
# INFERENCE PARAMETERS
# ============================================================================
INFERENCE_CONFIG = {
    "method": "monotonic_ranking",  # How to order the files
    "reverse": False,  # Whether to reverse the final order
}


# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================
OUTPUT_CONFIG = {
    "save_features": True,
    "save_plots": True,
    "experiment_name": "experiment_01",
}


# ============================================================================
# HELPER FUNCTION TO GET CURRENT ALGORITHM CLASS
# ============================================================================
def get_algorithm_class():
    """
    Returns the algorithm class based on ALGORITHM_NAME.
    
    Returns:
        Algorithm class to instantiate
    """
    # This will be implemented once we have algorithm classes
    algorithm_map = {
        # "signal_processing": SignalProcessingAlgorithm,
        # "statistical": StatisticalAlgorithm,
        # "ml_based": MLBasedAlgorithm,
    }
    
    if ALGORITHM_NAME not in algorithm_map:
        raise ValueError(
            f"Unknown algorithm: {ALGORITHM_NAME}. "
            f"Available algorithms: {list(algorithm_map.keys())}"
        )
    
    return algorithm_map[ALGORITHM_NAME]


def get_algorithm_config() -> Dict[str, Any]:
    """Returns the configuration for the current algorithm."""
    return ALGORITHM_CONFIG.get(ALGORITHM_NAME, {})
