
# File:         evaluation.py
# Assignment:   Linear Regression Architecture / MLOps Workshop
# Course:       CSCN8010 - Foundations of Machine Learning Frameworks
# Date:         September 25th, 2026
# Team 3:       Ricardo Mohammed (7500382), Senay Teweldebrhan (9120588)
#               Zeynep Ozdemir (9045142), Juan Camilo Chirivi (9115141)
# Description:  This file contains model evaluation metrics and experiment result
#               persistence logic for the linear regression project.


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import csv
from pathlib import Path


def get_next_experiment_id():
    """
    Get the next experiment ID from results.csv.

    This helper inspects the experiment log and returns the next sequential ID
    to store a new evaluation result.

    Returns:
        Integer representing the next experiment ID.

    Constraints:
        - The results file must contain an integer experiment_id column.
        - If the file is missing or empty, the function returns 1.
    """

    project_root = Path(__file__).resolve().parent.parent
    results_path = project_root / "experiments" / "results.csv"

    # If the file does not exist or only contains the header
    if not results_path.exists():
        return 1

    with open(results_path, "r") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return 1

    # Get the highest experiment ID and add 1
    last_id = max(int(row["experiment_id"]) for row in rows)

    return last_id + 1


def evaluate_model(model, X_test, y_test):
    """
    This function produces prediction metrics on the test set to assess model
    performance using RMSE, MAE, and R-squared.

    Parameters:
        model: Fitted regression model used for inference.
        X_test: Test feature matrix.
        y_test: Ground-truth test target values.

    Returns:
        Tuple containing RMSE, MAE, and R2 metric values.

    Constraints:
        - The model must already be trained before evaluation.
        - X_test and y_test must contain matching rows.
        - The target values must be numeric for metric calculation.

    Process:
    1. Use the trained model to make predictions.
    2. Compare predictions with the real values.
    3. Calculate RMSE, MAE, and R2.

    Returns:
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        r2: R-squared score
    """

    # Use the trained model to predict values from the test dataset.
    predictions = model.predict(X_test)

    # RMSE gives more importance to large prediction errors.
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # MAE calculates the average absolute difference
    # between predicted and actual values.
    mae = mean_absolute_error(y_test, predictions)

    # R2 measures how much of the target variation
    # is explained by the model.
    r2 = r2_score(y_test, predictions)

    # Send the three metrics back to orchestrator_main.py.
    return rmse, mae, r2


def save_experiment_results(
    experiment_id,
    dataset,
    feature,
    target,
    test_size,
    rmse,
    mae,
    r2,
    results_path
):
    """
    Save experiment metrics to experiments/results.csv.

    This helper appends a single experiment record containing the dataset,
    feature-target configuration, test size, and evaluation metrics.

    Parameters:
        experiment_id: Unique identifier for the experiment.
        dataset: Dataset name used for the experiment.
        feature: Feature column selected for the model.
        target: Target column predicted by the model.
        test_size: Proportion used for the validation split.
        rmse: Root Mean Squared Error value.
        mae: Mean Absolute Error value.
        r2: R-squared value.

    Constraints:
        - The results directory must exist before writing the CSV row.
        - The values must be numeric where required and ordered consistently.
    """

    # Find the root folder of the project.
    # evaluation.py is inside src/, so parent.parent goes to the project root.
    project_root = Path(__file__).resolve().parent.parent

    # Combine the project root with the path from experiment_config.yaml.
    # Example:
    # project_root + "experiments/results.csv"
    full_results_path = project_root / results_path

    # Open results.csv in append mode ("a").
    # Append means previous experiments are NOT deleted.
    with open(full_results_path, "a", newline="") as file:

        writer = csv.writer(file)

        # Add one new row containing the experiment information
        # and the model evaluation metrics.
        writer.writerow([
            experiment_id,
            dataset,
            feature,
            target,
            test_size,
            rmse,
            mae,
            r2
        ])


def get_next_experiment_id(results_path):
    """
    Find the next available experiment ID.

    Example:
        If results.csv contains experiments 1, 2, and 3,
        the next experiment ID will be 4.
    """

    # Find the project root folder.
    project_root = Path(__file__).resolve().parent.parent

    # Get the results.csv location from the configuration path.
    full_results_path = project_root / results_path

    # If results.csv does not exist yet,
    # this will be the first experiment.
    if not full_results_path.exists():
        return 1

    # Read the existing experiment history.
    with open(full_results_path, "r") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # If the file exists but does not contain experiments,
    # start with experiment ID 1.
    if not rows:
        return 1

    # Find the largest existing experiment ID.
    last_id = max(int(row["experiment_id"]) for row in rows)

    # The new experiment receives the next ID.
    return last_id + 1