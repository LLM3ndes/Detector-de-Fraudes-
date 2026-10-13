import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier


def train_fraud_model(
    X_train_res: np.ndarray,
    y_train_res: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> tuple[XGBClassifier, dict, np.ndarray, float, float]:
  """Treina o modelo XGBoost nos dados balanceados pelo SMOTE e avalia o desempenho nos dados de teste reais."""
  print("[+] Treinando o modelo XGBoost Classifier...")
  model = XGBClassifier(
      n_estimators=100,
      max_depth=5,
      learning_rate=0.1,
      eval_metric="logloss",
      random_state=random_state,
  )

  # Treinamento nos dados com SMOTE
  model.fit(X_train_res, y_train_res)

  # Predições e probabilidades no conjunto de teste original (desbalanceado)
  y_pred = model.predict(X_test)
  y_proba = model.predict_proba(X_test)[:, 1]

  # Cálculo das métricas de avaliação
  auc_roc = roc_auc_score(y_test, y_proba)
  precision, recall, _ = precision_recall_curve(y_test, y_proba)
  auc_pr = auc(recall, precision)
  cm = confusion_matrix(y_test, y_pred)
  report = classification_report(y_test, y_pred, output_dict=True)

  return model, report, cm, auc_roc, auc_pr