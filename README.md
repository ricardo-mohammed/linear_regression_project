# Linear Regression Architecture Workshop

## Project Summary

This project explores univariate linear regression using California and Ontario housing data. It covers data collection, model training, and evaluation. It also applies MLOps practices by organizing the code into modules, managing settings with a YAML file, and saving experiment results so they can be compared and reproduced.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Experiment settings are stored in `configs/experiment_config.yaml`. The current pipeline uses the California housing dataset, with `median_income` as the feature and `median_house_value` as the target.

## Run the Pipeline

From the project root directory, run:

```bash
python src/orchestrator_main.py
```

The pipeline loads and prepares the data, trains a linear regression model, and evaluates it using RMSE, MAE, and R². Each run saves its results to `experiments/results.csv`.

## Key Design Decisions

- Separate modules handle data loading, preprocessing, model training, and evaluation.
- A YAML file stores experiment settings so they can be changed without editing the pipeline code.
- A fixed random state makes the train/test split repeatable.
- Experiment metrics are saved to a CSV file so runs can be compared.