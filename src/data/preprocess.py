import pandas as pd


def process_na_values(df: pd.DataFrame):

    # Make a copy of dataframe
    df_processed = df.copy()

    # Remove columns with too much na values
    df_processed.drop(columns=['years_in_position', 'work_life_balance'], inplace=True)


    return df_processed


def process_categorical(df: pd.DataFrame):
    # Make a copy of dataframe
    df_processed = df.copy()

    # Drop over_18 and gender columns
    df_processed.drop(columns=['over_18', 'gender'], inplace=True)

    # Fill missing values for categorical columns
    df_processed['education'] = df_processed['education'].fillna('Bachelor')

    df_processed['job_satisfaction'] = df_processed['job_satisfaction'].fillna('High')

    df_processed['engagement'] = df_processed['engagement'].fillna('High')

    return df_processed