from data.load_data import load_data
from data.preprocess import preprocess_dataset
from features.build_features import feature_engineering_economic_costs
from features.prepare_features import prepare_features, train_test_split_data
from models.train_model import train_model
from models.predict_model import predict_model
from models.evaluate_model import evaluate_model
from models.visualize_model import visualize_tree, visualize_feature_importance
from config import RAW_EMPLOYEE_CHURN_FILE, PROCESSED_DATA_DIR, FIGURES_DIR


if __name__ == "__main__":
    # Load raw data
    df = load_data(RAW_EMPLOYEE_CHURN_FILE)

    # Preprocess data
    df_ml = preprocess_dataset(df)

    # Build features
    df_ml = feature_engineering_economic_costs(df_ml)

    # Prepare features and target variable
    #X, y, feature_names = prepare_features(df_ml)
    df_ml = prepare_features(df_ml)
    
    # Split data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split_data(df_ml, target='turnover', test_size=0.3)

    # Train model
    model = train_model(train_x, train_y)

    # Make predictions
    predictions, predictions_prob = predict_model(model, test_x)

    # Evaluate model
    metrics = evaluate_model(y_true=test_y, y_pred=predictions, y_proba=predictions_prob)
    print("Model Evaluation Metrics:", metrics['classification_report'])
    print("ROC AUC Score:", metrics['roc_auc_score'])

    # Visualize decision tree
    tree_fig = visualize_tree(model, feature_names=train_x.columns.tolist())
    tree_fig.savefig(FIGURES_DIR / "decision_tree.png")

    # Visualize feature importance
    importance_fig = visualize_feature_importance(model, feature_names=train_x.columns.tolist())
    importance_fig.savefig(FIGURES_DIR / "feature_importance.png")

    # Save results data
    df_ml['scoring_turnover_prob'] = model.predict_proba(df_ml.drop(columns=['turnover']))[:, 1]
    df_ml.to_csv(PROCESSED_DATA_DIR / "employee_churn_scored.csv", index=False)
