"""
Signal Processing Algorithm

Uses signal processing techniques and feature extraction to order files.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..base_algorithm import BaseAlgorithm
from ..feature_extractor import FeatureExtractor
from ..inference import OrderInference


class SignalProcessingAlgorithm(BaseAlgorithm):
    """
    Algorithm based on signal processing and statistical features.

    This algorithm:
    1. Extracts time and frequency domain features
    2. Uses monotonic ranking to order files by degradation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the signal processing algorithm.

        Args:
            config: Configuration dictionary with keys:
                - features: List of feature types to extract
                - frequency_bands: Fault band frequencies
                - sampling_rate: Sampling rate in Hz
        """
        super().__init__(config)

        self.sampling_rate = config.get("sampling_rate", 93750)
        self.feature_extractor = FeatureExtractor(self.sampling_rate)
        self.frequency_bands = config.get(
            "frequency_bands", [231, 3781, 5781, 4408]
        )
        self.selected_features = config.get("features", ["rms", "kurtosis"])

    def preprocess(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Minimal preprocessing - just pass through.

        More advanced preprocessing happens in the preprocessor module
        and should be done before calling the algorithm.

        Args:
            data: Dictionary of file_id -> DataFrame

        Returns:
            Same data (no preprocessing in this algorithm)
        """
        return data

    def extract_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract signal processing features from all files.

        Args:
            data: Dictionary of file_id -> DataFrame

        Returns:
            DataFrame of features indexed by file_id
        """
        features = self.feature_extractor.extract_features_from_all(
            data, fault_bands=self.frequency_bands
        )

        return features

    def infer_order(self, features: pd.DataFrame) -> np.ndarray:
        """
        Infer chronological order using monotonic ranking.

        Assumes degradation features increase monotonically over time.

        Args:
            features: DataFrame of features

        Returns:
            Array of file_ids in predicted chronological order
        """
        # Filter to selected features if specified
        if self.selected_features:
            available_features = [
                f for f in self.selected_features if f in features.columns
            ]

            if not available_features:
                print(
                    f"Warning: None of the selected features {self.selected_features} found."
                )
                print(f"Available features: {features.columns.tolist()}")
                print("Using all available features instead.")
                available_features = None
        else:
            available_features = None

        # Use monotonic ranking
        predicted_order = OrderInference.monotonic_ranking(
            features, feature_columns=available_features
        )

        return predicted_order
