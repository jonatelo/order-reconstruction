"""
Preprocessor Module

Handles data preprocessing operations.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import signal


class Preprocessor:
    """Preprocesses vibration signals."""

    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Preprocessing configuration dict
        """
        self.config = config
        self.normalize = config.get("normalize", True)
        self.detrend = config.get("detrend", False)
        self.remove_outliers = config.get("remove_outliers", False)
        self.filter_type = config.get("filter_type", None)
        self.filter_params = config.get("filter_params", {})

    def preprocess_file(
        self, df: pd.DataFrame, sampling_rate: float = 93750
    ) -> pd.DataFrame:
        """
        Preprocess a single file's data.

        Args:
            df: DataFrame with 'acceleration' and 'zct' columns
            sampling_rate: Sampling rate in Hz

        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()

        if "acceleration" not in df_processed.columns:
            return df_processed

        acceleration = df_processed["acceleration"].values

        # Remove outliers
        if self.remove_outliers:
            acceleration = self._remove_outliers(acceleration)

        # Detrend
        if self.detrend:
            acceleration = signal.detrend(acceleration)

        # Apply filter
        if self.filter_type:
            acceleration = self._apply_filter(acceleration, sampling_rate)

        # Normalize
        if self.normalize:
            acceleration = self._normalize(acceleration)

        df_processed["acceleration"] = acceleration

        return df_processed

    def preprocess_all(
        self, data: Dict[str, pd.DataFrame], sampling_rate: float = 93750
    ) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all files.

        Args:
            data: Dictionary of file_id -> DataFrame
            sampling_rate: Sampling rate in Hz

        Returns:
            Dictionary of preprocessed data
        """
        preprocessed = {}

        for file_id, df in data.items():
            preprocessed[file_id] = self.preprocess_file(df, sampling_rate)

        return preprocessed

    def _normalize(self, signal_data: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance."""
        mean = np.mean(signal_data)
        std = np.std(signal_data)

        if std > 0:
            return (signal_data - mean) / std
        return signal_data - mean

    def _remove_outliers(
        self, signal_data: np.ndarray, n_std: float = 5.0
    ) -> np.ndarray:
        """
        Remove outliers beyond n standard deviations.

        Args:
            signal_data: Input signal
            n_std: Number of standard deviations for outlier threshold

        Returns:
            Signal with outliers clipped
        """
        mean = np.mean(signal_data)
        std = np.std(signal_data)

        upper_bound = mean + n_std * std
        lower_bound = mean - n_std * std

        return np.clip(signal_data, lower_bound, upper_bound)

    def _apply_filter(
        self, signal_data: np.ndarray, sampling_rate: float
    ) -> np.ndarray:
        """
        Apply frequency domain filter.

        Args:
            signal_data: Input signal
            sampling_rate: Sampling rate in Hz

        Returns:
            Filtered signal
        """
        filter_type = self.filter_type
        params = self.filter_params

        if filter_type == "lowpass":
            cutoff = params.get("cutoff", 10000)
            order = params.get("order", 5)
            sos = signal.butter(
                order, cutoff, btype="low", fs=sampling_rate, output="sos"
            )
            return signal.sosfilt(sos, signal_data)

        elif filter_type == "highpass":
            cutoff = params.get("cutoff", 100)
            order = params.get("order", 5)
            sos = signal.butter(
                order, cutoff, btype="high", fs=sampling_rate, output="sos"
            )
            return signal.sosfilt(sos, signal_data)

        elif filter_type == "bandpass":
            low = params.get("low", 100)
            high = params.get("high", 10000)
            order = params.get("order", 5)
            sos = signal.butter(
                order, [low, high], btype="band", fs=sampling_rate, output="sos"
            )
            return signal.sosfilt(sos, signal_data)

        return signal_data
