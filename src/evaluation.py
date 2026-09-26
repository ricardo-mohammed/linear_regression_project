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
    Evaluate the trained Linear Regression model.

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
    """

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return rmse, mae, r2

def save_experiment_results(
    experiment_id,
    dataset,
    feature,
    target,
    test_size,
    rmse,
    mae,
    r2
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

    # Find project root
    project_root = Path(__file__).resolve().parent.parent

    # Path to results file
    results_path = project_root / "experiments" / "results.csv"

    # Open CSV and append a new experiment
    with open(results_path, "a", newline="") as file:
        writer = csv.writer(file)

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