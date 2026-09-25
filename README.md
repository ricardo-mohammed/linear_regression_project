

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration - Session 2

Experiment parameters are defined in:

```text
configs/experiment_config.yaml
```

The configuration controls the dataset, selected feature, target variable, train/test split, and other experiment parameters.

## Run the Pipeline

From the project root directory:

```bash
python src/orchestrator_main.py
```

The pipeline performs:

1. Data loading
2. Data preprocessing
3. Train/test splitting
4. Linear Regression training
5. Model evaluation
6. Experiment tracking

Model performance is evaluated using RMSE, MAE, and R².

Experiment results are automatically stored in:

```text
experiments/results.csv
```