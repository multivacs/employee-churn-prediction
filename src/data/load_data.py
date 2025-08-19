import pandas as pd
from pathlib import Path
from src.config import RAW_EMPLOYEE_CHURN_FILE


def load_data(path: Path = RAW_EMPLOYEE_CHURN_FILE, id_column: str = 'id', na_values: str = '#N/D'):
    return pd.read_csv(path, sep=',', decimal='.', index_col = id_column, na_values = na_values)