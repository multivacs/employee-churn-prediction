import pandas as pd
from pathlib import Path
from src.config import RAW_EMPLOYEE_CHURN_FILE


def load_data(path: Path = RAW_EMPLOYEE_CHURN_FILE):
    return pd.read_csv(path, sep=',', decimal='.', index_col= 'Employee ID')