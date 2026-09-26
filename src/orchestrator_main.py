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


# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

# Find the root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of the YAML configuration file.
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_config.yaml"


def load_config():
    """
    Load the experiment configuration from the YAML file.

    The YAML file contains parameters such as:
    - Dataset
    - Feature
    - Target
    - Train/test split
    - Random state
    - Experiment results path
    """

    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    return config


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

if __name__ == "__main__":

    # STEP 1 - Load experiment configuration
    config = load_config()
    print("Configuration loaded successfully!")

    # Read the main experiment parameters from the YAML file.
    dataset = config["data"]["dataset"]
    feature = config["data"]["selected_feature"]
    target = config["data"]["target"]
    results_path = config["experiment"]["results_path"]


    # STEP 2 - Check if the required configuration exists
    if dataset is None or feature is None or target is None:

        print("Pipeline is not ready yet.")
        print("Waiting for Session 1 configuration:")
        print(f"Dataset: {dataset}")
        print(f"Feature: {feature}")
        print(f"Target: {target}")

    else:

        print("Configuration is complete.")
        print("Pipeline is ready to run.")


        # STEP 3 - Build the dataset path
        dataset_path = config["data"]["raw_path"] + dataset


        # STEP 4 - Load the dataset
        df = load_csv(dataset_path)

        print(f"Data loaded successfully: {df.shape[0]} rows")


        # STEP 5 - Preprocess the data
        #
        # Select X and y, remove missing values,
        # and create the train/test split.
        X_train, X_test, y_train, y_test = preprocess_data(
            df,
            feature,
            target,
            test_size=config["training"]["test_size"],
            random_state=config["training"]["random_state"]
        )

        print("Data preprocessing completed.")


        # STEP 6 - Train the Linear Regression model
        model = train_model(X_train, y_train)

        print("Model training completed.")


        # STEP 7 - Evaluate the trained model
        #
        # Calculate RMSE, MAE and R2.
        rmse, mae, r2 = evaluate_model(
            model,
            X_test,
            y_test
        )

        print("Model evaluation completed.")

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.4f}")


        # STEP 8 - Save experiment results
        #
        # A new experiment ID is generated automatically.
        # The results are appended to results.csv.
        save_experiment_results(
            experiment_id=get_next_experiment_id(results_path),
            dataset=dataset,
            feature=feature,
            target=target,
            test_size=config["training"]["test_size"],
            rmse=rmse,
            mae=mae,
            r2=r2,
            results_path=results_path
        )

        print("Experiment results saved successfully!")