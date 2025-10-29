"""
Output Manager Module

Handles saving results, predictions, and visualizations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


class OutputManager:
    """Manages saving of results and outputs."""

    def __init__(self, results_dir: Path, experiment_name: str):
        """
        Initialize output manager.

        Args:
            results_dir: Directory to save results
            experiment_name: Name of the experiment
        """
        self.results_dir = Path(results_dir)
        self.experiment_name = experiment_name

        # Create experiment directory
        self.experiment_dir = self.results_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.plots_dir = self.experiment_dir / "plots"
        self.data_dir = self.experiment_dir / "data"

        self.plots_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        print(f"Results will be saved to: {self.experiment_dir}")

    def save_prediction(
        self, predicted_order: np.ndarray, filename: str = "submission.csv"
    ) -> Path:
        """
        Save prediction in submission format.

        Args:
            predicted_order: Array of file IDs in predicted chronological order
            filename: Output filename

        Returns:
            Path to saved file
        """
        # Create submission DataFrame
        # Format: each row is a file_id mapped to its position
        submission = pd.DataFrame(
            {
                "file_id": predicted_order,
                "position": range(len(predicted_order)),
            }
        )

        output_path = self.experiment_dir / filename
        submission.to_csv(output_path, index=False)

        print(f"✓ Saved prediction to: {output_path}")
        return output_path

    def save_features(
        self, features: pd.DataFrame, filename: str = "features.csv"
    ) -> Path:
        """
        Save extracted features.

        Args:
            features: DataFrame of features
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.data_dir / filename
        features.to_csv(output_path)

        print(f"✓ Saved features to: {output_path}")
        return output_path

    def save_metadata(
        self, metadata: Dict, filename: str = "metadata.json"
    ) -> Path:
        """
        Save experiment metadata.

        Args:
            metadata: Dictionary of metadata
            filename: Output filename

        Returns:
            Path to saved file
        """
        # Add timestamp
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["experiment_name"] = self.experiment_name

        output_path = self.experiment_dir / filename

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved metadata to: {output_path}")
        return output_path

    def save_plot(self, fig, filename: str) -> Path:
        """
        Save a matplotlib figure.

        Args:
            fig: Matplotlib figure object
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.plots_dir / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        print(f"✓ Saved plot to: {output_path}")
        return output_path

    def create_summary_report(
        self,
        predicted_order: np.ndarray,
        features: pd.DataFrame,
        algorithm_name: str,
        config: Dict,
        metrics: Optional[Dict] = None,
    ) -> Path:
        """
        Create a summary report of the experiment.

        Args:
            predicted_order: Predicted file order
            features: Feature DataFrame
            algorithm_name: Name of algorithm used
            config: Configuration dictionary
            metrics: Optional metrics dictionary

        Returns:
            Path to report file
        """
        report_lines = [
            "=" * 80,
            "ORDER RECONSTRUCTION EXPERIMENT REPORT",
            "=" * 80,
            f"Experiment: {self.experiment_name}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Algorithm: {algorithm_name}",
            "",
            "-" * 80,
            "CONFIGURATION",
            "-" * 80,
        ]

        # Add configuration
        for key, value in config.items():
            report_lines.append(f"  {key}: {value}")

        report_lines.extend(
            [
                "",
                "-" * 80,
                "RESULTS",
                "-" * 80,
                f"Number of files: {len(predicted_order)}",
                f"Number of features: {len(features.columns)}",
                "",
                "Feature names:",
            ]
        )

        for feat in features.columns:
            report_lines.append(f"  - {feat}")

        report_lines.extend(
            [
                "",
                "Predicted order (first 10):",
                f"  {predicted_order[:10]}",
                "",
            ]
        )

        # Add metrics if provided
        if metrics:
            report_lines.extend(
                [
                    "-" * 80,
                    "METRICS",
                    "-" * 80,
                ]
            )

            for metric_name, metric_value in metrics.items():
                report_lines.append(f"  {metric_name}: {metric_value}")

            report_lines.append("")

        report_lines.extend(["=" * 80, ""])

        # Save report
        output_path = self.experiment_dir / "report.txt"

        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"✓ Saved report to: {output_path}")
        return output_path
