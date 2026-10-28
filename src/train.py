"""Generate telecom data and train an explainable churn classifier."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/churn_model.joblib"
RANDOM_STATE = 42


def generate_data(rows=1500):
    """Create a deterministic, realistic dataset for a self-contained demo."""
    rng = np.random.default_rng(RANDOM_STATE)
    tenure = rng.integers(1, 73, rows)
    monthly_charges = rng.normal(70, 25, rows).clip(18, 140).round(2)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[.56, .24, .20])
    internet = rng.choice(["DSL", "Fiber optic", "None"], rows, p=[.34, .52, .14])
    support = rng.choice(["Yes", "No"], rows, p=[.42, .58])
    payment = rng.choice(["Electronic check", "Card", "Bank transfer"], rows, p=[.35, .35, .30])
    logit = (-1.1 + 1.35 * (contracts == "Month-to-month") + .75 * (internet == "Fiber optic")
             + .55 * (support == "No") + .45 * (payment == "Electronic check")
             + .018 * (monthly_charges - 70) - .035 * tenure)
    probability = 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "tenure_months": tenure, "monthly_charges": monthly_charges,
        "contract": contracts, "internet_service": internet,
        "tech_support": support, "payment_method": payment,
        "churn": (rng.random(rows) < probability).astype(int),
    })


def train():
    data = generate_data()
    X, y = data.drop(columns="churn"), data["churn"]
    numeric = ["tenure_months", "monthly_charges"]
    categorical = ["contract", "internet_service", "tech_support", "payment_method"]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("preprocess", preprocessor), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print(classification_report(y_test, predictions, target_names=["stay", "churn"]))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
