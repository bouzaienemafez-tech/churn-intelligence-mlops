# Churn Intelligence: Explainable ML API

A portfolio-grade, end-to-end machine learning project that predicts telecom customer churn and exposes predictions through a production-style REST API.

> Built to demonstrate the skills expected in an end-of-studies internship: data preparation, reproducible training, model evaluation, deployment, and interpretable business recommendations.

## Highlights

- Synthetic-but-realistic telecom dataset generator (no private data required)
- Scikit-learn preprocessing pipeline: imputation, encoding and scaling
- Cross-validated logistic-regression baseline with precision/recall reporting
- Probability-based churn-risk predictions and business actions
- FastAPI service with interactive Swagger documentation
- Dockerfile and structured project layout

## Architecture

```
data generation → training pipeline → model artifact → FastAPI prediction service
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/train.py
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and use `/predict`. The service gives a churn probability, a risk level, and a retention suggestion.

## Project structure

```
app/main.py         FastAPI inference API
src/train.py        Reproducible data generation and training
models/             Generated model artifact (ignored by Git)
requirements.txt    Python dependencies
Dockerfile          Containerised deployment
```

## Evaluation approach

Churn is an imbalanced classification problem, so the training output reports more than accuracy: ROC-AUC, precision, recall, and F1 score. A real next step would be to connect the pipeline to a customer database, track experiments with MLflow, and add automated model monitoring.

## Skills demonstrated

`Python` · `Pandas` · `Scikit-learn` · `Feature Engineering` · `Classification` · `Model Evaluation` · `FastAPI` · `Docker` · `MLOps`
