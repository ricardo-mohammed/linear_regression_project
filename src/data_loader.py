# Pandas is used to read the CSV file and convert it
# into a DataFrame that can be processed by the pipeline.
import pandas as pd

# Path is used to build file paths relative to the project root.
# This helps avoid hard-coded absolute paths.
from pathlib import Path


def load_csv(file_path):
    """
    Load a CSV dataset and return it as a Pandas DataFrame.

    Process:
    1. Find the root directory of the project.
    2. Combine the project root with the relative CSV path.
    3. Read the CSV file using Pandas.
    4. Return the dataset as a DataFrame.

    Parameters:
        file_path: Relative path to the CSV dataset.

    Returns:
        df: Pandas DataFrame containing the loaded dataset.
    """

    # data_loader.py is located inside src/.
    # parent.parent moves from src/ to the project root.
    project_root = Path(__file__).resolve().parent.parent

    # Create the complete path using the project root
    # and the relative path received from the configuration.
    #
    # Example:
    # project_root + "data/raw/california_housing.csv"
    csv_path = project_root / file_path

    # Read the CSV file and convert it into a DataFrame.
    df = pd.read_csv(csv_path)

    # Return the DataFrame to orchestrator_main.py.
    return df