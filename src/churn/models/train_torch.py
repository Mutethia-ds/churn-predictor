"""Train a small PyTorch MLP for churn prediction, tracked with MLflow.

This is an alternative model to the scikit-learn one in train_sklearn.py,
useful for demonstrating a deep-learning workflow in the same project.
"""

from __future__ import annotations

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from churn.config import PROJECT_ROOT, get_secrets, get_settings
from churn.features.build_features import build_feature_table


class ChurnMLP(nn.Module):
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_training_arrays():
    settings = get_settings()
    raw_path = PROJECT_ROOT / settings["data"]["raw_path"]
    df = pd.read_csv(raw_path)
    feature_df = build_feature_table(df, settings)

    target_col = settings["data"]["target_column"]
    drop_cols = [target_col, "customer_id"]

    X = feature_df.drop(
        columns=[c for c in drop_cols if c in feature_df.columns]
    ).astype("float32")

    y = feature_df[target_col].astype("float32")

    return X, y, settings


def train() -> str:
    secrets = get_secrets()
    torch.manual_seed(secrets.random_seed)

    X, y, settings = load_training_arrays()
    cfg = settings["model"]["torch"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=secrets.random_seed,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    model = ChurnMLP(
        n_features=X_train.shape[1],
        hidden_size=cfg["hidden_size"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
    )

    loss_fn = nn.BCEWithLogitsLoss()

    mlflow.set_tracking_uri(secrets.mlflow_tracking_uri)
    mlflow.set_experiment(settings["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="pytorch_mlp"):

        mlflow.log_params(cfg)

        model.train()

        for epoch in range(cfg["epochs"]):
            epoch_loss = 0.0

            for xb, yb in train_loader:
                optimizer.zero_grad()

                logits = model(xb)

                loss = loss_fn(logits, yb)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(train_ds)

            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        model.eval()

        with torch.no_grad():
            logits = model(
                torch.tensor(X_test_scaled, dtype=torch.float32)
            )

            proba = torch.sigmoid(logits).numpy()

        preds = (proba >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        mlflow.log_metrics(metrics)

        print("Metrics:", metrics)

        model_dir = PROJECT_ROOT / "models"
        model_dir.mkdir(exist_ok=True)

        torch.save(
            model.state_dict(),
            model_dir / "torch_model.pt",
        )

        np.save(
            model_dir / "scaler_mean.npy",
            scaler.mean_,
        )

        np.save(
            model_dir / "scaler_scale.npy",
            scaler.scale_,
        )

        # -------- MLflow 3.x compatible model logging --------
        input_example = torch.tensor(
            X_train_scaled[:5],
            dtype=torch.float32,
        )

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            input_example=input_example,
            serialization_format="pickle",
        )
        # -----------------------------------------------------

        run_id = mlflow.active_run().info.run_id

        print(f"MLflow run: {run_id}")

        return run_id


if __name__ == "__main__":
    train()