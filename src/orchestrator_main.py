# File:         orchestrator_main.py
# Assignment:   Linear Regression Architecture / MLOps Workshop
# Course:       CSCN8010 - Foundations of Machine Learning Frameworks
# Date:         September 25th, 2026
# Team 3:       Ricardo Mohammed (7500382), Senay Teweldebrhan (9120588)
#               Zeynep Ozdemir (9045142), Juan Camilo Chirivi (9115141)
# Description:  This file orchestrates the data loading, preprocessing, model
#               training, evaluation, and experiment logging pipeline.


import yaml
from pathlib import Path

from data_loader import load_csv
from preprocessing import preprocess_data
from model import train_model
from evaluation import (
    evaluate_model,
    save_experiment_results,
    get_next_experiment_id
)
# Find the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the YAML configuration file
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.yaml"


def load_config():
    """
    Load experiment configuration from the YAML file.

    This helper reads the project configuration and returns the structured
    dictionary used to drive the training pipeline.

    Returns:
        Python dictionary containing the experiment configuration values.

    Constraints:
        - The configuration file must exist at the expected project path.
        - The YAML content must include the required data and training keys.
    """

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":

    # Load configuration
    config = load_config()

    print("Configuration loaded successfully!")

    # Get dataset configuration
    dataset = config["data"]["dataset"]
    feature = config["data"]["selected_feature"]
    target = config["data"]["target"]

    # Check if Session 1 configuration is available
    if dataset is None or feature is None or target is None:
        print("Pipeline is not ready yet.")
        print("Waiting for Session 1 configuration:")
        print(f"Dataset: {dataset}")
        print(f"Feature: {feature}")
        print(f"Target: {target}")

    else:
        print("Configuration is complete.")
        print("Pipeline is ready to run.")

    # Build the dataset path from the configuration
    dataset_path = config["data"]["raw_path"] + dataset

    # 1. Load data
    df = load_csv(dataset_path)
    print(f"Data loaded successfully: {df.shape[0]} rows")

    # 2. Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        feature,
        target,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )
    print("Data preprocessing completed.")

    # 3. Train model
    model = train_model(X_train, y_train)
    print("Model training completed.")

    # 4. Evaluate model
    rmse, mae, r2 = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("Model evaluation completed.")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")

        # 5. Save experiment results
    save_experiment_results(
        experiment_id=get_next_experiment_id(),
        dataset=dataset,
        feature=feature,
        target=target,
        test_size=config["training"]["test_size"],
        rmse=rmse,
        mae=mae,
        r2=r2
    )

    print("Experiment results saved successfully!")