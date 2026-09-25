from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import csv
from pathlib import Path

def get_next_experiment_id():
    """
    Get the next experiment ID from results.csv.
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

    Returns:
        RMSE, MAE and R2 metrics.
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