import pandas as pd
from pathlib import Path
from src.config import RAW_EMPLOYEE_CHURN_FILE


def load_data(path: Path):
    return pd.read_csv(RAW_EMPLOYEE_CHURN_FILE, sep=',', decimal='.')