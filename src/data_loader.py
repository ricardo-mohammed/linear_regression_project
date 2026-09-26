# File:         data_loader.py
# Assignment:   Linear Regression Architecture / MLOps Workshop
# Course:       CSCN8010 - Foundations of Machine Learning Frameworks
# Date:         September 25th, 2026
# Team 3:       Ricardo Mohammed (7500382), Senay Teweldebrhan (9120588)
#               Zeynep Ozdemir (9045142), Juan Camilo Chirivi (9115141)
# Description:  This file contains the utility for loading CSV data into a pandas
#               DataFrame for the machine learning pipeline.


import pandas as pd
from pathlib import Path


def load_csv(file_path):
    """
    Load a CSV file and return it as a pandas DataFrame.

    This function resolves the input path relative to the project root and reads
    the CSV file using pandas.

    Parameters:
        file_path: Relative path to the CSV file from the project root.

    Returns:
        Pandas DataFrame containing the loaded dataset.

    Constraints:
        - The path must point to a valid CSV file in the project structure.
        - The file must exist before the function is called.
        - The file content must follow a valid tabular CSV format.
    """

    # Find the root directory of the project
    project_root = Path(__file__).resolve().parent.parent

    # Create the complete path to the CSV
    csv_path = project_root / file_path

    # Load the CSV
    df = pd.read_csv(csv_path)

    return df


def load_raw_datasets():
    """
    Load the raw California and Ontario housing datasets.

    This helper resolves the project root and reads the raw CSV files used in the
    exploratory regression analysis and model workflow.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: California dataset and Ontario
        dataset, respectively.

    Constraints:
        - The files must exist under the project data/raw directory.
        - Both CSV files must contain valid tabular data and expected columns.
        - The function assumes the default project folder layout is preserved.
    """

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"

    california_df = pd.read_csv(raw_dir / "california_housing.csv")
    ontario_df = pd.read_csv(raw_dir / "ontario_housing_2024.csv")

    return california_df, ontario_df