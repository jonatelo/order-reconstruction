"""
Feature Extractor Module

Provides common signal processing and feature extraction methods.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


class FeatureExtractor:
    """Extracts features from vibration signals."""

    def __init__(self, sampling_rate: float = 93750):
        """
        Initialize feature extractor.

        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate

    def extract_time_domain_features(
        self, acceleration: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract time-domain statistical features.

        Args:
            acceleration: Acceleration time series

        Returns:
            Dictionary of feature name -> value
        """
        features = {
            "mean": np.mean(acceleration),
            "std": np.std(acceleration),
            "rms": np.sqrt(np.mean(acceleration**2)),
            "peak": np.max(np.abs(acceleration)),
            "peak_to_peak": np.ptp(acceleration),
            "crest_factor": np.max(np.abs(acceleration))
            / np.sqrt(np.mean(acceleration**2)),
            "skewness": stats.skew(acceleration),
            "kurtosis": stats.kurtosis(acceleration),
            "energy": np.sum(acceleration**2),
        }

        return features

    def extract_frequency_domain_features(
        self, acceleration: np.ndarray, fault_bands: List[float] = None
    ) -> Dict[str, float]:
        """
        Extract frequency-domain features including FFT-based metrics.

        Args:
            acceleration: Acceleration time series
            fault_bands: List of fault band center frequencies in Hz

        Returns:
            Dictionary of feature name -> value
        """
        # Compute FFT
        n = len(acceleration)
        fft_vals = np.fft.fft(acceleration)
        fft_freq = np.fft.fftfreq(n, 1 / self.sampling_rate)

        # Only positive frequencies
        pos_mask = fft_freq > 0
        fft_freq = fft_freq[pos_mask]
        fft_mag = np.abs(fft_vals[pos_mask])

        features = {
            "spectral_mean": np.mean(fft_mag),
            "spectral_std": np.std(fft_mag),
            "spectral_peak": np.max(fft_mag),
            "peak_frequency": fft_freq[np.argmax(fft_mag)],
            "spectral_energy": np.sum(fft_mag**2),
        }

        # Spectral entropy
        psd = fft_mag**2 / np.sum(fft_mag**2)
        psd = psd[psd > 0]  # Remove zeros for log
        features["spectral_entropy"] = -np.sum(psd * np.log2(psd))

        # Fault band power features
        if fault_bands:
            for i, center_freq in enumerate(fault_bands):
                band_power = self._compute_band_power(
                    fft_freq, fft_mag, center_freq, bandwidth=100
                )
                features[f"fault_band_{i + 1}_power"] = band_power

        return features

    def extract_tachometer_features(self, zct: np.ndarray) -> Dict[str, float]:
        """
        Extract features from tachometer zero-cross timestamps.

        Args:
            zct: Zero-cross timestamps

        Returns:
            Dictionary of feature name -> value
        """
        # Remove NaN values
        zct_clean = zct[~np.isnan(zct)]

        if len(zct_clean) < 2:
            return {
                "rpm_mean": 0,
                "rpm_std": 0,
                "rpm_trend": 0,
            }

        # Compute time differences between zero crossings
        dt = np.diff(zct_clean)

        # Convert to RPM (revolutions per minute)
        # dt is in samples, convert to seconds then to RPM
        dt_seconds = dt / self.sampling_rate
        rpm = 60 / dt_seconds

        features = {
            "rpm_mean": np.mean(rpm),
            "rpm_std": np.std(rpm),
            "rpm_min": np.min(rpm),
            "rpm_max": np.max(rpm),
            "rpm_range": np.ptp(rpm),
        }

        # Linear trend in RPM over time
        if len(rpm) > 1:
            x = np.arange(len(rpm))
            slope, _, _, _, _ = stats.linregress(x, rpm)
            features["rpm_trend"] = slope
        else:
            features["rpm_trend"] = 0

        return features

    def extract_all_features(
        self, df: pd.DataFrame, fault_bands: List[float] = None
    ) -> Dict[str, float]:
        """
        Extract all features from a single file.

        Args:
            df: DataFrame with 'acceleration' and 'zct' columns
            fault_bands: List of fault band center frequencies

        Returns:
            Dictionary of all features
        """
        features = {}

        # Time-domain features
        if "acceleration" in df.columns:
            acceleration = df["acceleration"].values
            time_features = self.extract_time_domain_features(acceleration)
            features.update(time_features)

            # Frequency-domain features
            freq_features = self.extract_frequency_domain_features(
                acceleration, fault_bands
            )
            features.update(freq_features)

        # Tachometer features
        if "zct" in df.columns:
            zct = df["zct"].values
            tacho_features = self.extract_tachometer_features(zct)
            features.update(tacho_features)

        return features

    def extract_features_from_all(
        self, data: Dict[str, pd.DataFrame], fault_bands: List[float] = None
    ) -> pd.DataFrame:
        """
        Extract features from all files.

        Args:
            data: Dictionary of file_id -> DataFrame
            fault_bands: List of fault band center frequencies

        Returns:
            DataFrame with features, indexed by file_id
        """
        feature_list = []

        for file_id, df in data.items():
            features = self.extract_all_features(df, fault_bands)
            features["file_id"] = file_id
            feature_list.append(features)

        feature_df = pd.DataFrame(feature_list)
        feature_df = feature_df.set_index("file_id")

        return feature_df

    def _compute_band_power(
        self,
        freq: np.ndarray,
        magnitude: np.ndarray,
        center_freq: float,
        bandwidth: float = 100,
    ) -> float:
        """
        Compute power in a frequency band.

        Args:
            freq: Frequency array
            magnitude: Magnitude array
            center_freq: Center frequency of the band
            bandwidth: Bandwidth around center frequency (±bandwidth/2)

        Returns:
            Total power in the band
        """
        lower = center_freq - bandwidth / 2
        upper = center_freq + bandwidth / 2

        band_mask = (freq >= lower) & (freq <= upper)
        band_power = np.sum(magnitude[band_mask] ** 2)

        return band_power
