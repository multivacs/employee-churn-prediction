import pandas as pd
from math import ceil
import matplotlib.pyplot as plt

def visualize_categorical(categoricals: pd.Series):
    """
    Generate bar plots for categorical features in a DataFrame.
    """

    # Determine the number of rows and columns for the subplots
    n = categoricals.shape[1]
    ncols = 2
    nrows = ceil(n / ncols)

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    # Plot each categorical feature
    for i, col in enumerate(categoricals):
        categoricals[col].value_counts().plot(kind='barh', ax=axes[i])
        axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Frequency')
        axes[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()