import numpy as np
import pandas as pd

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