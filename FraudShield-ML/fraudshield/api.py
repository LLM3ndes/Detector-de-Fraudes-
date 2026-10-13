from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd

# Instância do servidor FastAPI
app = FastAPI(
    title="FraudShield-ML API",
    description="API de Detecção e Classificação de Fraudes Financeiras em Tempo Real",
    version="1.0.0",
)

# Esquema de validação dos dados de entrada via Pydantic


class TransactionInput(BaseModel):
  V1: float
  V2: float
  V3: float
  V4: float
  V5: float
  amount: float


# Armazenamento global em memória do modelo e do scaler
model_global = None
scaler_global = None


def set_model_and_scaler(model, scaler):
  """Injeta o modelo treinado e o scaler na API em tempo de execução."""
  global model_global, scaler_global
  model_global = model
  scaler_global = scaler


@app.get("/")
def health_check():
  """Endpoint de verificação de integridade do servidor."""
  return {
      "status": "online",
      "system": "FraudShield-ML API",
      "model_loaded": model_global is not None,
  }


@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
  """Endpoint principal para classificar uma transação em tempo real."""
  if model_global is None or scaler_global is None:
    raise HTTPException(
        status_code=500,
        detail=(
            "O modelo de Machine Learning não foi carregado na memória da API."
        ),
    )

  # Converte o payload JSON para um DataFrame do Pandas
  input_df = pd.DataFrame([{
      "V1": transaction.V1,
      "V2": transaction.V2,
      "V3": transaction.V3,
      "V4": transaction.V4,
      "V5": transaction.V5,
      "amount": transaction.amount,
  }])

  # Normaliza a transação usando o mesmo Scaler do treino
  scaled_data = scaler_global.transform(input_df)

  # Executa a predição e probabilidade
  prediction = int(model_global.predict(scaled_data)[0])
  probability = float(model_global.predict_proba(scaled_data)[0][1])

  # Classificação de Nível de Risco
  if probability > 0.70:
    risk_level = "ALTO"
  elif probability > 0.30:
    risk_level = "MÉDIO"
  else:
    risk_level = "BAIXO"

  return {
      "is_fraud": bool(prediction == 1),
      "fraud_probability": round(probability * 100, 2),
      "risk_level": risk_level,
      "recommendation": (
          "BLOQUEAR TRANSAÇÃO" if prediction == 1 else "APROVAR TRANSAÇÃO"
      ),
  }