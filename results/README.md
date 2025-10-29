# Experiment Results

This directory contains results from different experimental approaches.

## Structure

Each experiment creates a subdirectory named by `experiment_name` from `config.py`:

```
results/
├── experiment_01/
│   ├── submission.csv          # Predicted order
│   ├── report.txt              # Summary report
│   ├── metadata.json           # Configuration used
│   ├── data/
│   │   └── features.csv        # Extracted features
│   └── plots/
│       └── feature_trends.png  # Visualizations
├── experiment_02/
│   └── ...
```

## Submission Format

The `submission.csv` file contains:
- `file_id`: The file identifier
- `position`: The predicted chronological position (0 = earliest)

This format can be used for leaderboard submissions.
