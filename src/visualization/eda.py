import pandas as pd
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

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


def statistics_numerical(num: pd.Series):
    """
    Generate descriptive statistics for numerical features in a DataFrame.
    """

    statistics = num.describe().T
    # Add median to statistics
    statistics['median'] = num.median()
    # Change order to get the median next to the mean
    statistics = statistics[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]

    return statistics


def analysis_by_target(df: pd.DataFrame, target: str, feature: str, normalize = True):
    """
    Analyze the relationship between a feature and the target variable.
    """

    analysis = df.groupby(feature)[target].agg(['mean', 'count']).reset_index()
    analysis.rename(columns={'mean': f'{target}_mean', 'count': 'total'}, inplace=True)
    analysis = analysis.sort_values(by=f'{target}_mean', ascending=False)

    if normalize:
        analysis[f'{target}_mean'] = round(analysis[f'{target}_mean'] * 100, 2)
        y_title = f'{target} Mean (%)'
    else:
        y_title = f'{target} Mean'

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.bar(analysis[feature], analysis[f'{target}_mean'], color='skyblue')
    plt.xlabel(feature)
    plt.ylabel(y_title)
    plt.title(f'{target} Mean by {feature}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return analysis


def feature_engineering_economic_costs(df: pd.DataFrame):
    """
    Create new features related to employee turnover costs.
    """

    # Create annual salary feature
    df['annual_salary'] = df['monthly_salary'] * 12

    # Define economic impact constants
    conditions = [(df['annual_salary'] < 30000),
                  (df['annual_salary'] >= 30000) & (df['annual_salary'] < 50000),
                  (df['annual_salary'] >= 50000) & (df['annual_salary'] < 75000),
                  (df['annual_salary'] >= 75000)]
    
    costs = [df.annual_salary * 0.161, # 16.1% for <30k
             df.annual_salary * 0.197, # 19.7% for 30k-50k
             df.annual_salary * 0.204, # 20.4% for 50k-75k
             df.annual_salary * 0.21] # 21% for >=75k
    
    df['cost_of_turnover'] = np.select(conditions, costs, default=0)

    return df
