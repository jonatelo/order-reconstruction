"""
Algorithms package for order reconstruction.

Each algorithm should inherit from BaseAlgorithm and implement:
- preprocess()
- extract_features()
- infer_order()
"""

from .signal_processing import SignalProcessingAlgorithm

__all__ = ['SignalProcessingAlgorithm']
