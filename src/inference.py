"""
Inference Module

Provides methods for inferring chronological order from features.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class OrderInference:
    """Methods for inferring chronological order from degradation features."""
    
    @staticmethod
    def monotonic_ranking(features: pd.DataFrame, 
                         feature_columns: List[str] = None,
                         reverse: bool = False) -> np.ndarray:
        """
        Infer order assuming monotonic increase in degradation features.
        
        Args:
            features: DataFrame of features indexed by file_id
            feature_columns: List of feature columns to use. If None, uses all numeric columns
            reverse: If True, assumes features decrease over time
        
        Returns:
            Array of file_ids in predicted chronological order
        """
        if feature_columns is None:
            feature_columns = features.select_dtypes(include=[np.number]).columns.tolist()
        
        # Compute aggregate degradation score
        # Normalize each feature to [0, 1] and take mean
        normalized = features[feature_columns].copy()
        
        for col in feature_columns:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            
            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0
        
        # Aggregate score (mean of normalized features)
        degradation_score = normalized.mean(axis=1)
        
        # Sort by degradation score
        sorted_indices = degradation_score.sort_values(ascending=not reverse).index
        
        return np.array(sorted_indices)
    
    @staticmethod
    def pca_based_ordering(features: pd.DataFrame,
                          feature_columns: List[str] = None) -> np.ndarray:
        """
        Use PCA to find the principal degradation direction.
        
        Args:
            features: DataFrame of features indexed by file_id
            feature_columns: List of feature columns to use
        
        Returns:
            Array of file_ids in predicted chronological order
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        if feature_columns is None:
            feature_columns = features.select_dtypes(include=[np.number]).columns.tolist()
        
        X = features[feature_columns].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled)
        
        # Sort by first principal component
        sorted_indices = np.argsort(pc1.flatten())
        file_ids = features.index[sorted_indices]
        
        return np.array(file_ids)
    
    @staticmethod
    def weighted_ranking(features: pd.DataFrame,
                        feature_weights: Dict[str, float]) -> np.ndarray:
        """
        Rank files using weighted combination of features.
        
        Args:
            features: DataFrame of features indexed by file_id
            feature_weights: Dictionary mapping feature name -> weight
        
        Returns:
            Array of file_ids in predicted chronological order
        """
        # Normalize features
        normalized = features.copy()
        
        for feature_name in feature_weights.keys():
            if feature_name not in normalized.columns:
                continue
            
            min_val = normalized[feature_name].min()
            max_val = normalized[feature_name].max()
            
            if max_val > min_val:
                normalized[feature_name] = (normalized[feature_name] - min_val) / (max_val - min_val)
            else:
                normalized[feature_name] = 0
        
        # Compute weighted score
        weighted_score = pd.Series(0, index=features.index)
        
        for feature_name, weight in feature_weights.items():
            if feature_name in normalized.columns:
                weighted_score += weight * normalized[feature_name]
        
        # Sort by weighted score
        sorted_indices = weighted_score.sort_values().index
        
        return np.array(sorted_indices)
    
    @staticmethod
    def compute_spearman_footrule(predicted_order: np.ndarray,
                                  true_order: np.ndarray) -> float:
        """
        Compute Spearman footrule distance between predicted and true order.
        
        The footrule distance is the sum of absolute differences in ranks.
        
        Args:
            predicted_order: Predicted sequence of file IDs
            true_order: True sequence of file IDs
        
        Returns:
            Spearman footrule distance (0 = perfect match)
        """
        # Create rank mappings
        predicted_ranks = {file_id: rank for rank, file_id in enumerate(predicted_order)}
        true_ranks = {file_id: rank for rank, file_id in enumerate(true_order)}
        
        # Compute footrule distance
        distance = 0
        for file_id in predicted_ranks.keys():
            distance += abs(predicted_ranks[file_id] - true_ranks[file_id])
        
        return distance
    
    @staticmethod
    def compute_spearman_correlation(predicted_order: np.ndarray,
                                    true_order: np.ndarray) -> float:
        """
        Compute Spearman rank correlation between predicted and true order.
        
        Args:
            predicted_order: Predicted sequence of file IDs
            true_order: True sequence of file IDs
        
        Returns:
            Spearman correlation coefficient (1 = perfect correlation)
        """
        # Create rank mappings
        predicted_ranks = {file_id: rank for rank, file_id in enumerate(predicted_order)}
        true_ranks = {file_id: rank for rank, file_id in enumerate(true_order)}
        
        # Get ranks in same order
        pred_ranks_list = [predicted_ranks[fid] for fid in predicted_ranks.keys()]
        true_ranks_list = [true_ranks[fid] for fid in predicted_ranks.keys()]
        
        correlation, _ = spearmanr(pred_ranks_list, true_ranks_list)
        
        return correlation
