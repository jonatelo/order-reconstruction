# Order Reconstruction Challenge

## Overview

This repository contains experimental approaches to solve the TII (Technology Innovation Institute) Order Reconstruction Coding Challenge - a competition focused on reconstructing the chronological order of scrambled vibration data from a degrading bearing system.

## Challenge Summary

**Objective**: Reconstruct the true chronological sequence of scrambled vibration recordings from a helicopter turboshaft engine bearing degradation test.

**The Problem**: 
- A dataset of N high-frequency vibration and tachometer recordings has been intentionally shuffled
- Files are named `file_1.csv` through `file_N.csv` but are in random order
- The task is to determine the true chronological order by analyzing degradation patterns

**Why It Matters**: Bearings degrade gradually over time, and this degradation is reflected in vibration signals. Despite operational variability making the data stochastic, degradation is unidirectional - bearings don't heal themselves. Detecting these patterns is crucial for predictive maintenance in aviation and other critical industries.

## Dataset Details

- **Vibration Data**: Acceleration time series sampled at 93,750 Hz (single channel)
- **Tachometer Data**: Zero-cross timestamps (zct)
- **Nominal turbine speed**: ~536.27 Hz
- **Bearing fault-band centers**: [231, 3781, 5781, 4408] Hz
- **File size**: ~55 MB zipped, significantly larger unzipped

Each file contains a short time-history of high-frequency vibration and tachometer data from an operating aircraft.

## Evaluation Metric

**Spearman Footrule Distance**: Measures how far the predicted file order is from the true chronological order.
- Perfect reconstruction = 0
- Leaderboard split: 50% public / 50% private

## Prize Pool

- **Total**: $26,000 USD
- **1st Place**: $12,000 USD
- **Runners-up**: $9,000 USD (shared by up to 5 teams)
- **Bonus**: $5,000 USD for top performer across both TII Coding Challenges

## Possible Approaches

This repository will explore various methodologies:

1. **Signal Processing**
   - Feature extraction from vibration signals
   - Frequency domain analysis
   - Fault-band amplitude tracking

2. **Statistical Degradation Modeling**
   - Monotonic trend detection
   - Statistical feature evolution

3. **Machine Learning / AI**
   - Sequence reconstruction algorithms
   - Deep learning for pattern recognition
   - Time-series ordering models

## Repository Structure

```
order-reconstruction/
├── data/                  # Dataset (not tracked in git)
├── notebooks/             # Exploratory analysis and experiments
├── src/                   # Source code
│   ├── algorithms/        # Algorithm implementations
│   ├── base_algorithm.py  # Abstract base class for algorithms
│   ├── config.py          # Configuration file (EDIT THIS!)
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocessor.py    # Preprocessing functions
│   ├── feature_extractor.py # Feature engineering
│   ├── inference.py       # Order inference methods
│   ├── output_manager.py  # Results saving
│   ├── utils.py           # Helper functions
│   └── main.py            # Main pipeline script
├── results/               # Submission files and results
└── requirements.txt       # Python dependencies
```

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Download the dataset** from the challenge page and place the CSV files in the `data/` directory.

3. **Configure your experiment** by editing `src/config.py`:
   - Set `ALGORITHM_NAME` to choose which algorithm to use
   - Adjust algorithm-specific parameters in `ALGORITHM_CONFIG`
   - Set preprocessing options in `PREPROCESSING_CONFIG`
   - Choose your experiment name in `OUTPUT_CONFIG`

4. **Run the pipeline:**
   ```bash
   uv run python -m src.main
   ```

## How It Works

The pipeline follows these steps:

1. **Load Data**: Reads CSV files from the `data/` directory
2. **Preprocess**: Applies filtering, normalization, etc. based on config
3. **Extract Features**: Computes time/frequency domain features
4. **Infer Order**: Uses the configured algorithm to predict chronological order
5. **Save Results**: Outputs submission file, features, plots, and report

All results are saved to `results/<experiment_name>/`.

## Creating New Algorithms

To implement a new algorithm:

1. Create a new file in `src/algorithms/`, e.g., `my_algorithm.py`
2. Inherit from `BaseAlgorithm` and implement:
   - `preprocess()` - Custom preprocessing
   - `extract_features()` - Feature extraction
   - `infer_order()` - Order reconstruction logic
3. Add your algorithm to `src/algorithms/__init__.py`
4. Update `src/config.py` to include your algorithm in the imports and config

Example structure:
```python
from src.base_algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def preprocess(self, data):
        # Your preprocessing logic
        return preprocessed_data
    
    def extract_features(self, data):
        # Your feature extraction
        return features_df
    
    def infer_order(self, features):
        # Your ordering logic
        return predicted_order
```

## Experimentation Workflow

1. **Exploratory Analysis**: Use notebooks in `notebooks/` to explore data and test ideas
2. **Implement Algorithm**: Create algorithm class in `src/algorithms/`
3. **Configure**: Set parameters in `src/config.py`
4. **Run Pipeline**: Execute `python -m src.main`
5. **Evaluate**: Check results in `results/<experiment_name>/`
6. **Iterate**: Adjust config and repeat

## Challenge Link

[TII Order Reconstruction Challenge](https://tii.community.innocentive.com/challenge/861e8f75ece541b99a9cc4b1ea9866bb)

## Notes

- Each experiment is saved with a unique name - change `experiment_name` in config for each run
- The `SignalProcessingAlgorithm` is provided as a baseline implementation
- Focus on features that evolve monotonically over time (degradation doesn't reverse)
- The evaluation metric is Spearman footrule distance (lower is better)
