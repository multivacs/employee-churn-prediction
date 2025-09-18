from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

def visualize_tree(model, feature_names, figsize=(50, 50)):
    """
    Visualize a decision tree model.

    Parameters:
    - model: Trained decision tree model.
    - feature_names: List of feature names used in the model.
    - figsize: Tuple specifying the size of the figure.

    Returns:
    - fig: Matplotlib figure object containing the tree visualization.
    """

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(model, feature_names=feature_names, filled=True, impurity=False, node_ids=False, rounded=True, precision=2, ax=ax)
    plt.close(fig)  # Prevents display in non-interactive environments
    return fig

def visualize_feature_importance(model, feature_names, figsize=(10, 6)):
    """
    Visualize feature importance of a model.

    Parameters:
    - model: Trained model with feature_importances_ attribute.
    - feature_names: List of feature names used in the model.
    - figsize: Tuple specifying the size of the figure.

    Returns:
    - fig: Matplotlib figure object containing the feature importance visualization.
    """

    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df = importance_df[importance_df['Importance'] > 0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance')
    fig.tight_layout()
    plt.close(fig)  # Prevents display in non-interactive environments
    return fig