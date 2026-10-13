import os
import sys
import numpy as np
import pandas as pd
import uvicorn

# Adiciona o diretório base ao sys.path para importações no Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
  sys.path.insert(0, BASE_DIR)

from data.generate_fraud_dataset import generate_synthetic_fraud_data
from fraudshield.api import app, set_model_and_scaler
from fraudshield.model import train_fraud_model
from fraudshield.preprocessing import load_and_preprocess_data


def run_pipeline() -> None:
  print("=" * 70)
  print(
      "🛡️ FRAUDSHIELD-ML: SISTEMA DE DETECÇÃO DE FRAUDES BANCÁRIAS (XGBOOST +"
      " SMOTE)"
  )
  print("=" * 70)

  dataset_path = os.path.join(BASE_DIR, "data", "creditcard_transactions.csv")

  # 1. Verifica/Gera Dataset Sintético
  if not os.path.exists(dataset_path):
    print(
        "\n[!] Dataset não encontrado. Gerando dados sintéticos de"
        " transações..."
    )
    generate_synthetic_fraud_data()

  # 2. Carrega e Pré-processa Dados (SMOTE + StandardScaler)
  print("\n[1/3] Executando Pipeline de Pré-processamento e SMOTE...")
  X_train_res, y_train_res, X_test, y_test, scaler = load_and_preprocess_data(
      dataset_path
  )

  # 3. Treina e Avalia o Modelo XGBoost
  print("\n[2/3] Treinando Algoritmo XGBoost Classifier...")
  model, report, cm, auc_roc, auc_pr = train_fraud_model(
      X_train_res, y_train_res, X_test, y_test
  )

  # Acesso seguro às métricas da classe de fraude (chave '1' ou 1)
  fraud_metrics = report.get("1", report.get(1, {}))
  recall_fraud = fraud_metrics.get("recall", 0.0) * 100
  precision_fraud = fraud_metrics.get("precision", 0.0) * 100

  print("\n📊 RELATÓRIO DE AVALIAÇÃO DO MODELO (DADOS DE TESTE REAIS):")
  print(f" • ROC-AUC Score : {auc_roc:.4f}")
  print(
      " • PR-AUC Score : "
      f"{auc_pr:.4f} (Métrica Chave para Dados Desbalanceados)"
  )
  print(f" • Acurácia Geral : {report['accuracy']*100:.1f}%")
  print(
      " • Revocação (Recall - Fraudes) :"
      f" {recall_fraud:.1f}% (Detecção de Fraudes)"
  )
  print(f" • Precisão (Precision - Fraudes): {precision_fraud:.1f}%")

  print("\n🧩 Matriz de Confusão:")
  print(
      f" • Legítimas Aprovadas (TN): {cm[0][0]:>4} | Falsos Alarmes (FP):"
      f" {cm[0][1]:>4}"
  )
  print(
      f" • Fraudes Não Detectadas (FN): {cm[1][0]:>4} | Fraudes Bloqueadas"
      f" (TP): {cm[1][1]:>4}"
  )

  # 4. Injeta Modelo e Scaler na API FastAPI
  print("\n[3/3] Carregando Inteligência Preditiva na API FastAPI...")
  set_model_and_scaler(model, scaler)

  print("\n" + "=" * 70)
  print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
  print("=" * 70)
  print("🚀 Iniciando Servidor Web FastAPI em http://127.0.0.1:8000 ...")
  print("💡 Pressione 'Ctrl + C' no terminal para encerrar o servidor.")
  print("=" * 70 + "\n")

  # Inicia o servidor HTTP em background via Uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
  run_pipeline()