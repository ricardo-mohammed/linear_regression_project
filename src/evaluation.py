# Import the evaluation metrics used to measure model performance.
# MSE is used to calculate RMSE.
# MAE measures the average absolute prediction error.
# R2 measures how well the model explains the variation in the target.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# NumPy is used here to calculate the square root for RMSE.
import numpy as np

# CSV is used to save each experiment result in results.csv.
import csv

# Path helps create file paths that work independently
# of the folder where the program is executed.
from pathlib import Path


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained Linear Regression model.

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
    Save the results of one experiment into a CSV file.

    The location of results.csv is received from the YAML configuration
    instead of being hard-coded in this file.
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