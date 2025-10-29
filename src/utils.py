"""
Utility functions and constants for the order reconstruction project.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def extract_file_id(filename: str) -> str:
    """
    Extract numeric file ID from filename.
    
    Args:
        filename: e.g., "file_5.csv" or "file_5"
    
    Returns:
        File ID as string, e.g., "5"
    """
    basename = filename.replace('.csv', '')
    parts = basename.split('_')
    
    if len(parts) >= 2:
        return parts[1]
    
    return basename


def create_submission_format(predicted_order: np.ndarray) -> pd.DataFrame:
    """
    Create submission DataFrame in the required format.
    
    Args:
        predicted_order: Array of file IDs in chronological order
    
    Returns:
        DataFrame ready for submission
    """
    return pd.DataFrame({
        'file_id': predicted_order,
        'rank': range(1, len(predicted_order) + 1)
    })


def normalize_features(features: pd.DataFrame, 
                      method: str = 'standard') -> pd.DataFrame:
    """
    Normalize feature values.
    
    Args:
        features: DataFrame of features
        method: 'standard' (z-score), 'minmax' (0-1), or 'robust'
    
    Returns:
        Normalized features DataFrame
    """
    normalized = features.copy()
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        # Z-score normalization
        for col in numeric_cols:
            mean = features[col].mean()
            std = features[col].std()
            
            if std > 0:
                normalized[col] = (features[col] - mean) / std
    
    elif method == 'minmax':
        # Min-Max normalization to [0, 1]
        for col in numeric_cols:
            min_val = features[col].min()
            max_val = features[col].max()
            
            if max_val > min_val:
                normalized[col] = (features[col] - min_val) / (max_val - min_val)
    
    elif method == 'robust':
        # Robust scaling using median and IQR
        for col in numeric_cols:
            median = features[col].median()
            q75 = features[col].quantile(0.75)
            q25 = features[col].quantile(0.25)
            iqr = q75 - q25
            
            if iqr > 0:
                normalized[col] = (features[col] - median) / iqr
    
    return normalized


def visualize_feature_trends(features: pd.DataFrame, 
                            predicted_order: np.ndarray,
                            feature_cols: List[str] = None):
    """
    Create visualization of feature trends over predicted order.
    
    Args:
        features: DataFrame of features
        predicted_order: Predicted chronological order
        feature_cols: List of features to plot (None = all numeric)
    
    Returns:
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    
    if feature_cols is None:
        feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Reorder features by predicted order
    ordered_features = features.loc[predicted_order, feature_cols]
    
    # Create subplots
    n_features = len(feature_cols)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        values = ordered_features[feature].values
        
        ax.plot(values, marker='o', linestyle='-', linewidth=2, markersize=4)
        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Time Order', fontsize=10)
        ax.set_ylabel('Feature Value', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def compute_bearing_fault_frequencies(turbine_speed: float,
                                     bearing_factors: Dict[str, float]) -> Dict[str, float]:
    """
    Compute bearing fault frequencies based on turbine speed.
    
    Args:
        turbine_speed: Turbine speed in Hz
        bearing_factors: Dictionary with keys: cage, ball, inner_race, outer_race
    
    Returns:
        Dictionary of fault frequencies in Hz
    """
    return {
        'cage': turbine_speed * bearing_factors['cage'],
        'ball': turbine_speed * bearing_factors['ball'],
        'inner_race': turbine_speed * bearing_factors['inner_race'],
        'outer_race': turbine_speed * bearing_factors['outer_race'],
    }


def print_order_summary(predicted_order: np.ndarray, 
                       features: pd.DataFrame = None):
    """
    Print a summary of the predicted order.
    
    Args:
        predicted_order: Array of file IDs
        features: Optional features DataFrame for additional info
    """
    print("\n" + "=" * 60)
    print("PREDICTED CHRONOLOGICAL ORDER")
    print("=" * 60)
    print(f"Total files: {len(predicted_order)}")
    print(f"\nFirst 10 files in order: {predicted_order[:10]}")
    print(f"Last 10 files in order: {predicted_order[-10:]}")
    
    if features is not None:
        print("\nFeature statistics along predicted order:")
        print("-" * 60)
        
        ordered_features = features.loc[predicted_order]
        numeric_cols = ordered_features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Show first 5 features
            values = ordered_features[col].values
            trend = "↑" if values[-1] > values[0] else "↓"
            print(f"{col:20s}: {values[0]:.4f} → {values[-1]:.4f} {trend}")
    
    print("=" * 60 + "\n")
