import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(
    csv_path: str, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, StandardScaler]:
  """Carrega o dataset, realiza train/test split estratificado, normaliza as variáveis e aplica SMOTE na base de treino."""
  if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Arquivo não encontrado em: {csv_path}")

  df = pd.read_csv(csv_path)
  X = df.drop(columns=["is_fraud"])
  y = df["is_fraud"]

  # 1. Divisão Treino/Teste estratificada (preserva a % de fraudes nos dois conjuntos)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state, stratify=y
  )

  # 2. Padronização / Escalonamento (StandardScaler)
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # 3. Aplicação do SMOTE APENAS no conjunto de treino (evita Data Leakage)
  smote = SMOTE(random_state=random_state)
  X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

  print("[+] Dados divididos com sucesso!")
  print(
      f" • Treino original : {len(X_train)} amostras (Fraudes:"
      f" {(y_train == 1).sum()})"
  )
  print(
      f" • Treino com SMOTE : {len(X_train_res)} amostras (Fraudes:"
      f" {(y_train_res == 1).sum()})"
  )
  print(
      f" • Teste (Sem alteração): {len(X_test)} amostras (Fraudes:"
      f" {(y_test == 1).sum()})"
  )

  return X_train_res, y_train_res, X_test_scaled, y_test, scaler