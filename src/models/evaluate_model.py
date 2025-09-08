from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

def evaluate_model(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series):
    """
    Evaluate the model's performance using classification report and ROC AUC score.
    """

    report = classification_report(y_true, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_true, y_proba)

    evaluation = {
        'classification_report': pd.DataFrame(report).transpose(),
        'roc_auc_score': roc_auc
    }

    return evaluation