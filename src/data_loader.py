import pandas as pd
from pathlib import Path


def load_csv(file_path):
    """
    Load a CSV file and return it as a Pandas DataFrame.
    """

    # Find the root directory of the project
    project_root = Path(__file__).resolve().parent.parent

    # Create the complete path to the CSV
    csv_path = project_root / file_path

    # Load the CSV
    df = pd.read_csv(csv_path)

    return df