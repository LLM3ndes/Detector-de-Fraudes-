# 🛡️ FraudShield-ML: Sistema de Detecção de Fraudes Financeiras com Machine Learning & FastAPI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EB5757?style=for-the-badge&logo=xgboost)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Um framework robusto de Ciência de Dados e Engenharia de Machine Learning projetado para a detecção em tempo real de fraudes em transações bancárias. O sistema lida com o desafio crítico de **dados altamente desbalanceados** utilizando técnicas avançadas de reamostragem (SMOTE) e algoritmos de *boosting* de alta performance (XGBoost), servidos através de uma API de baixa latência em FastAPI.

---

## 📌 Principais Funcionalidades & Módulos

O projeto é estruturado em uma arquitetura modular orientada a produção:

1. **Geração de Dados Sintéticos (`data/generate_fraud_dataset.py`)**
   - Cria um dataset realista de transações bancárias com proporção severamente desbalanceada (padrão de fraude em torno de 3%).
   - Simula variáveis estatísticas latentes (`V1` a `V5`) e valores monetários modelados por distribuições exponenciais distintas para transações legítimas e fraudulentas.

2. **Pré-processamento e Mitigação de Desbalanceamento (`fraudshield/preprocessing.py`)**
   - Realiza divisão estratificada de treino e teste para preservar a taxa de fraudes em ambos os conjuntos.
   - Aplica padronização de variáveis com `StandardScaler`.
   - Utiliza **SMOTE (Synthetic Minority Over-sampling Technique)** exclusivamente no conjunto de treinamento para evitar *data leakage*, balanceando as classes de forma sintética.

3. **Treinamento e Avaliação Preditiva (`fraudshield/model.py`)**
   - Treina um classificador **XGBoost** otimizado para classificação binária.
   - Avalia métricas essenciais para cenários de fraudes: **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, **Matriz de Confusão**, **Recall (Revocação)** e **Precision**.

4. **API de Inferência em Tempo Real (`fraudshield/api.py`)**
   - Construída com **FastAPI** e validada com **Pydantic**.
   - Expõe um endpoint `/predict` para classificar transações instantaneamente, retornando a probabilidade de fraude, nível de risco (*Baixo*, *Médio*, *Alto*) e recomendação automatizada (*Aprovar* ou *Bloquear*).

5. **Orquestrador de Pipeline (`main.py`)**
   - Automatiza o fluxo completo: verificação/geração de dados, pré-processamento, SMOTE, treinamento, avaliação de métricas de negócio e inicialização automática do servidor HTTP Uvicorn.

---

## 📂 Arquitetura do Projeto

```text
FraudShield-ML/
├── main.py                     # Orquestrador do pipeline de ML & Inicializador da API
├── requirements.txt            # Dependências do projeto
├── README.md                   # Documentação do repositório
├── .gitignore                  # Arquivos ignorados pelo controle de versão
├── data/
│   ├── generate_fraud_dataset.py # Gerador de transações bancárias sintéticas
│   └── creditcard_transactions.csv # Dataset bruto gerado
└── fraudshield/
    ├── __init__.py             # Inicialização do pacote & versão
    ├── preprocessing.py        # Pipeline de limpeza, escalonamento e SMOTE
    ├── model.py                # Treinamento e avaliação do XGBoost Classifier
    └── api.py                  # Servidor Web FastAPI e endpoints de inferência
