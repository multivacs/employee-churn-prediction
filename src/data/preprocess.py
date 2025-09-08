import pandas as pd



def preprocess_dtypes(df: pd.DataFrame):
    """
    Preprocess the DataFrame by converting columns to appropriate dtypes.
    """
    df_processed = df.copy()
    df_processed['gender'] = df_processed['gender'].astype('object')
    df_processed['turnover'] = df_processed['turnover'].map({'Yes': 1, 'No': 0}).astype('int8')

    return df_processed


def preprocess_drop_columns(df: pd.DataFrame):
    """
    Preprocess the DataFrame by dropping columns with too many missing values or constant values.
    """

    # Make a copy of dataframe
    df_processed = df.copy()

    # Remove columns with too much na values
    df_processed.drop(columns=['years_in_position', 'work_life_balance'], inplace=True)

    # Remove constant columns
    df_processed.drop(columns=['employees', 'over_18', 'biweekly_hours'], inplace=True)

    # Remove columns inconsistent
    df_processed.drop(columns=['gender'], inplace=True)

    return df_processed


def preprocess_fill_na_values(df: pd.DataFrame):
    """
    Preprocess the DataFrame by filling missing values.
    """

    # Make a copy of dataframe
    df_processed = df.copy()


    # Fill missing values for categorical columns
    df_processed['education'] = df_processed['education'].fillna('Bachelor')

    df_processed['job_satisfaction'] = df_processed['job_satisfaction'].fillna('High')

    df_processed['engagement'] = df_processed['engagement'].fillna('High')

    return df_processed


def preprocess_dataset(df: pd.DataFrame):
    """
    Preprocess the DataFrame by applying all preprocessing steps.
    """
    # Remove duplicates
    df_processed = df.drop_duplicates().reset_index(drop=True)

    df_processed = preprocess_dtypes(df_processed)
    df_processed = preprocess_drop_columns(df_processed)
    df_processed = preprocess_fill_na_values(df_processed)
    return df_processed