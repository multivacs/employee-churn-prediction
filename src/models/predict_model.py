import pandas as pd

def predict_model(model, X: pd.DataFrame):
    """
    Generate predictions using the trained model.
    """

    predictions = model.predict_proba(X)[:, 1]  # Probability of the positive class
    return predictions