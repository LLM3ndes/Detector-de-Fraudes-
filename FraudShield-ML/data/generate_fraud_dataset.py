import os
import random
import numpy as np
import pandas as pd


def generate_synthetic_fraud_data(
    num_records: int = 5000, fraud_ratio: float = 0.03
) -> None:
  """Gera um dataset sintético realista de transações bancárias com dados altamente desbalanceados."""
  np.random.seed(42)
  random.seed(42)

  num_frauds = int(num_records * fraud_ratio)
  num_legit = num_records - num_frauds

  # Atributos de transações legítimas (valores menores, padrão V1-V5 centrado em zero)
  amount_legit = np.random.exponential(scale=80, size=num_legit) + 5.0
  v_legit = np.random.normal(loc=0.0, scale=1.0, size=(num_legit, 5))

  # Atributos de transações fraudulentas (valores maiores e anomalias estatísticas)
  amount_fraud = np.random.exponential(scale=350, size=num_frauds) + 50.0
  v_fraud = np.random.normal(loc=2.5, scale=1.5, size=(num_frauds, 5))

  # Unificação dos dados
  amounts = np.concatenate([amount_legit, amount_fraud])
  v_features = np.vstack([v_legit, v_fraud])
  is_fraud = np.concatenate([np.zeros(num_legit), np.ones(num_frauds)])

  df = pd.DataFrame(v_features, columns=[f"V{i+1}" for i in range(5)])
  df["amount"] = np.round(amounts, 2)
  df["is_fraud"] = is_fraud.astype(int)

  # Embaralha os registros
  df = df.sample(frac=1, random_state=42).reset_index(drop=True)

  # Salva o CSV na pasta data/
  script_dir = os.path.dirname(os.path.abspath(__file__))
  output_path = os.path.join(script_dir, "creditcard_transactions.csv")
  df.to_csv(output_path, index=False)

  print(f"[+] Dataset de fraudes gerado com sucesso em: {output_path}")
  print(
      f"[+] Total de registros: {len(df)} | Legítimas:"
      f" {(df['is_fraud'] == 0).sum()} | Fraudes: {(df['is_fraud'] == 1).sum()}"
      f" ({fraud_ratio*100:.1f}%)"
  )


if __name__ == "__main__":
  generate_synthetic_fraud_data()