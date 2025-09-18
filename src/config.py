"""
config.py
Central configuration for the Employee Churn project.
"""

from pathlib import Path


# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data files
RAW_EMPLOYEE_CHURN_FILE = RAW_DATA_DIR / "employee_churn_dataset.csv"

# Figures directory
FIGURES_DIR = BASE_DIR / "reports" / "figures"