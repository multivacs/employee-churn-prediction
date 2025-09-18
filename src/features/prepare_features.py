from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_features(df: pd.DataFrame):
    """
    Prepare features for modeling.
    """

    # One-hot encode categorical features
    cat_features = df.select_dtypes(include=['object']).columns
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    cat_encoded = ohe.fit_transform(df[cat_features])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_features))

    # Combine encoded categorical features to replace the original categorical features on DataFrame
    df = pd.concat([df, cat_encoded_df], axis=1)
    df.drop(cat_features, axis=1, inplace=True)

    return df


def train_test_split_data(df: pd.DataFrame, target: str, test_size: float = 0.3):
    """
    Split the DataFrame into training and testing sets.
    """

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=42)