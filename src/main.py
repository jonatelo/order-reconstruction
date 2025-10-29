"""
Main entry point for the Order Reconstruction pipeline.

Usage:
    python -m src.main
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.algorithms.signal_processing import SignalProcessingAlgorithm
from src.data_loader import DataLoader
from src.output_manager import OutputManager
from src.preprocessor import Preprocessor
from src.utils import print_order_summary, visualize_feature_trends


def main():
    """Main pipeline execution."""

    print("=" * 80)
    print("ORDER RECONSTRUCTION PIPELINE")
    print("=" * 80)
    print(f"Experiment: {config.OUTPUT_CONFIG['experiment_name']}")
    print(f"Algorithm: {config.ALGORITHM_NAME}")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("[1/5] Loading data...")
    data_loader = DataLoader(config.DATA_DIR)

    try:
        # Load all files (or set limit for testing)
        data = data_loader.load_all(limit=None)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure the data files are in the 'data/' directory.")
        return

    print(f"✓ Loaded {len(data)} files\n")

    # ========================================================================
    # 2. PREPROCESS DATA
    # ========================================================================
    print("[2/5] Preprocessing data...")
    preprocessor = Preprocessor(config.PREPROCESSING_CONFIG)
    preprocessed_data = preprocessor.preprocess_all(
        data, sampling_rate=config.DATASET_CONFIG["sampling_rate"]
    )
    print("✓ Preprocessing complete\n")

    # ========================================================================
    # 3. RUN ALGORITHM
    # ========================================================================
    print(f"[3/5] Running {config.ALGORITHM_NAME} algorithm...")

    # Get algorithm configuration
    algo_config = config.get_algorithm_config()
    algo_config["sampling_rate"] = config.DATASET_CONFIG["sampling_rate"]
    algo_config["frequency_bands"] = config.DATASET_CONFIG["fault_band_centers"]

    # Initialize algorithm
    algorithm = config.get_algorithm_class()(algo_config)

    # Run the algorithm
    predicted_order, features = algorithm.run(preprocessed_data)

    print("✓ Algorithm complete\n")

    # ========================================================================
    # 4. SAVE RESULTS
    # ========================================================================
    print("[4/5] Saving results...")
    output_manager = OutputManager(
        config.RESULTS_DIR, config.OUTPUT_CONFIG["experiment_name"]
    )

    # Save prediction
    output_manager.save_prediction(predicted_order)

    # Save features if requested
    if config.OUTPUT_CONFIG["save_features"]:
        output_manager.save_features(features)

    # Save metadata
    metadata = {
        "algorithm": config.ALGORITHM_NAME,
        "algorithm_config": algo_config,
        "preprocessing_config": config.PREPROCESSING_CONFIG,
        "n_files": len(data),
        "n_features": len(features.columns),
        "feature_names": features.columns.tolist(),
    }
    output_manager.save_metadata(metadata)

    # Create summary report
    output_manager.create_summary_report(
        predicted_order=predicted_order,
        features=features,
        algorithm_name=config.ALGORITHM_NAME,
        config=algo_config,
    )

    print()

    # ========================================================================
    # 5. VISUALIZE (optional)
    # ========================================================================
    if config.OUTPUT_CONFIG["save_plots"]:
        print("[5/5] Creating visualizations...")

        try:
            fig = visualize_feature_trends(features, predicted_order)
            output_manager.save_plot(fig, "feature_trends.png")
        except Exception as e:
            print(f"⚠ Warning: Could not create plots: {e}")
    else:
        print("[5/5] Skipping visualizations (disabled in config)")

    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_order_summary(predicted_order, features)

    print("\n✅ Pipeline completed successfully!")
    print(f"Results saved to: {output_manager.experiment_dir}")
    print()


if __name__ == "__main__":
    main()
