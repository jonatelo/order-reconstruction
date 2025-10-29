"""
Data Loader Module

Handles reading CSV files from the data directory.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class DataLoader:
    """Loads vibration and tachometer data from CSV files."""

    def __init__(self, data_dir: Path):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to directory containing data files
        """
        self.data_dir = Path(data_dir) / "files"

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

    def get_file_list(self, pattern: str = "file_*.csv") -> List[Path]:
        """
        Get list of data files matching the pattern.

        Args:
            pattern: Glob pattern for file matching

        Returns:
            List of file paths sorted by name
        """
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files found matching '{pattern}' in {self.data_dir}"
            )

        return files

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a single CSV file.

        Expected format:
        - acceleration: float column
        - zct: float column (tachometer zero-cross timestamps)

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with the file contents
        """
        try:
            df = pd.read_csv(file_path)

            # Validate expected columns
            # column v to acceleration:
            df = df.rename(columns={"v": "acceleration"})
            expected_cols = ["acceleration", "zct"]
            if not all(col in df.columns for col in expected_cols):
                print(
                    f"Warning: Expected columns {expected_cols} not found in {file_path}"
                )
                print(f"Available columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {str(e)}")

    def load_all(
        self, file_pattern: str = "file_*.csv", limit: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all data files into memory.

        Args:
            file_pattern: Glob pattern for file matching
            limit: Optional limit on number of files to load (for testing)

        Returns:
            Dictionary mapping file_id -> DataFrame
            file_id is the numeric ID extracted from filename (e.g., "file_5.csv" -> "5")
        """
        files = self.get_file_list(file_pattern)

        if limit:
            files = files[:limit]
            print(f"Loading {limit} files (limited for testing)")

        data = {}

        for file_path in files:
            # Extract file ID from filename (e.g., "file_5.csv" -> "5")
            file_id = file_path.stem.split("_")[1]

            print(f"Loading {file_path.name}...", end=" ")
            df = self.load_file(file_path)
            data[file_id] = df
            print(f"✓ ({len(df)} samples)")

        print(f"\nLoaded {len(data)} files successfully")
        return data

    def get_file_info(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get summary information about loaded files.

        Args:
            data: Dictionary of loaded data

        Returns:
            DataFrame with file statistics
        """
        info = []

        for file_id, df in data.items():
            info.append(
                {
                    "file_id": file_id,
                    "n_samples": len(df),
                    "has_acceleration": "acceleration" in df.columns,
                    "has_zct": "zct" in df.columns,
                    "acceleration_mean": df["acceleration"].mean()
                    if "acceleration" in df.columns
                    else None,
                    "acceleration_std": df["acceleration"].std()
                    if "acceleration" in df.columns
                    else None,
                }
            )

        return pd.DataFrame(info)
