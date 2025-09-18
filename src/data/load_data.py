import pandas as pd
from pathlib import Path
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import RAW_EMPLOYEE_CHURN_FILE


def load_data(path: Path = RAW_EMPLOYEE_CHURN_FILE, id_column: str = 'id', na_values: str = '#N/D'):
    return pd.read_csv(path, sep=',', decimal='.', index_col=id_column, na_values=na_values)