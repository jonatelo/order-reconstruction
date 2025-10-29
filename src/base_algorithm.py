"""
Base Algorithm Interface

All algorithm implementations should inherit from this base class
and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class BaseAlgorithm(ABC):
    """
    Abstract base class for order reconstruction algorithms.
    
    Each algorithm should implement:
    1. preprocess() - Data preprocessing specific to the algorithm
    2. extract_features() - Feature engineering from raw signals
    3. infer_order() - Reconstruct the chronological order
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the algorithm with configuration.
        
        Args:
            config: Dictionary containing algorithm-specific parameters
        """
        self.config = config
        self.features = None
        self.predicted_order = None
        
    @abstractmethod
    def preprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess the raw data according to algorithm requirements.
        
        Args:
            data: Dictionary mapping file_id -> DataFrame with columns:
                  ['acceleration', 'zct']
        
        Returns:
            Dictionary with preprocessed data
        """
        pass
    
    @abstractmethod
    def extract_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract features from the preprocessed data.
        
        Args:
            data: Dictionary of preprocessed data from preprocess()
        
        Returns:
            DataFrame with features, indexed by file_id
            Columns represent different features
        """
        pass
    
    @abstractmethod
    def infer_order(self, features: pd.DataFrame) -> np.ndarray:
        """
        Infer the chronological order based on extracted features.
        
        Args:
            features: DataFrame of features from extract_features()
        
        Returns:
            Array of file indices in chronological order
            e.g., [3, 1, 5, 2, 4] means file_3 is first, file_1 is second, etc.
        """
        pass
    
    def run(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Run the complete pipeline: preprocess -> extract features -> infer order.
        
        Args:
            data: Dictionary mapping file_id -> raw DataFrame
        
        Returns:
            Tuple of (predicted_order, features_dataframe)
        """
        print(f"[{self.__class__.__name__}] Starting preprocessing...")
        preprocessed_data = self.preprocess(data)
        
        print(f"[{self.__class__.__name__}] Extracting features...")
        self.features = self.extract_features(preprocessed_data)
        
        print(f"[{self.__class__.__name__}] Inferring order...")
        self.predicted_order = self.infer_order(self.features)
        
        return self.predicted_order, self.features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Optional: Return feature importance scores if applicable.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {}
