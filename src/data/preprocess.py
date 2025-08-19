import pandas as pd


def process_na_values(df: pd.DataFrame):

    # Make a copy of dataframe
    df_processed = df.copy()

    # Remove columns with too much na values
    df_processed.drop(columns=['years_in_position', 'work_life_balance'], inplace=True)


    return df_processed