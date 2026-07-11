"""Train a scikit-learn gradient boosting classifier for churn prediction,
tracking parameters, metrics, and the model itself with MLflow.
"""

from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from churn.config import PROJECT_ROOT, get_secrets, get_settings
from churn.features.build_features import build_feature_table


def load_training_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    settings = get_settings()
    raw_path = PROJECT_ROOT / settings["data"]["raw_path"]
    df = pd.read_csv(raw_path)
    feature_df = build_feature_table(df, settings)

    target_col = settings["data"]["target_column"]
    drop_cols = [target_col, "customer_id"]
    X = feature_df.drop(columns=[c for c in drop_cols if c in feature_df.columns])
    y = feature_df[target_col]
    return X, y, settings


def train() -> str:
    secrets = get_secrets()
    X, y, settings = load_training_data()
    cfg = settings["model"]["sklearn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=secrets.random_seed, stratify=y
    )

    mlflow.set_tracking_uri(secrets.mlflow_tracking_uri)
    mlflow.set_experiment(settings["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="gradient_boosting"):
        mlflow.log_params(cfg)

        model = GradientBoostingClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=secrets.random_seed,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        print("Metrics:", metrics)

        # Confusion matrix + ROC curve artifacts
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        model_dir = PROJECT_ROOT / "models"
        model_dir.mkdir(exist_ok=True)
        mlflow.sklearn.log_model(model, artifact_path="model")

        local_model_path = model_dir / "sklearn_model"
        if local_model_path.exists():
            shutil.rmtree(local_model_path)
        mlflow.sklearn.save_model(model, path=str(local_model_path))

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run: {run_id}")
        return run_id


if __name__ == "__main__":
    train()
