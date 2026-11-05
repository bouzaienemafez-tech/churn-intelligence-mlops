# Churn Intelligence — Classification API Demo

A **synthetic-data portfolio project** demonstrating a scikit-learn preprocessing pipeline, logistic-regression evaluation and FastAPI inference for a telecom-style churn scenario.

**Business question:** which customers might merit retention follow-up? The API assigns risk bands and fixed suggestions. It does not estimate the effect of a retention intervention.

## What is implemented

| Component | Evidence and scope |
| --- | --- |
| Data generator | [src/train.py](src/train.py): 1,500 synthetic rows, NumPy seed 42; six input features and a generated churn label |
| Preprocessing | Median/mode imputation, numeric scaling and one-hot encoding fitted within a pipeline |
| Classifier | Logistic regression with balanced class weights |
| Evaluation | One stratified 80/20 holdout split, seed 42; ROC-AUC and classification report |
| API | [app/main.py](app/main.py): health check, churn score, risk band and rule-based suggestion |
| Container definition | [Dockerfile](Dockerfile): trains during image build; container execution remains unverified |

```text
Seeded synthetic data → stratified split → preprocessing + logistic regression
                                              ├→ holdout metrics
                                              └→ saved pipeline → FastAPI
```

There is **no cross-validation, explanation endpoint, experiment tracker, model registry or monitoring** in the current repository. Logistic regression is inspectable, but feature contributions are not presented by this API. The repository name reflects an MLOps learning direction; this is not a production-ready system.

## Run locally

Verified with Python 3.13.7 on macOS ARM64. The Dockerfile specifies Python 3.12.
No dataset download or credentials are needed. Run from the repository root:

```bash
git clone https://github.com/bouzaienemafez-tech/churn-intelligence-mlops.git
cd churn-intelligence-mlops
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
python -m uvicorn app.main:app --reload
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.
Open http://127.0.0.1:8000/docs. Training writes `models/churn_model.joblib` (ignored by Git); the generated dataset exists in memory and is not exported.
Missing models produce HTTP 503. `/health` checks file existence only.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"tenure_months":8,"monthly_charges":89.5,"contract":"Month-to-month","internet_service":"Fiber optic","tech_support":"No","payment_method":"Electronic check"}'
```

Observed response in the verified environment:

```json
{"churn_probability":0.94,"risk_level":"high","recommendation":"High priority: offer a retention call and a tailored discount."}
```

Generator categories: contract = `Month-to-month`, `One year`, `Two year`; internet = `DSL`, `Fiber optic`, `None`; support = `Yes`, `No`; payment = `Electronic check`, `Card`, `Bank transfer`.
The API accepts other strings, which the encoder ignores; categorical validation is future work.

## Results and interpretation

Observed on the 300-row synthetic holdout (202 stay, 98 churn):

| Metric | Result |
| --- | --- |
| ROC-AUC | **0.772** |
| Churn precision | 0.54 |
| Churn recall | 0.70 |
| Churn F1 | 0.61 |
| Accuracy | 0.71 |

Training uses 1,200 rows. Classification metrics use the model's default decision threshold; API risk bands use 0.40 and 0.70. Those bands have not been optimized against retention costs. Class weighting means the returned score should not be treated as a calibrated real-world probability.

The labels come from a hand-written logistic formula using the input features. This exercise demonstrates the workflow, not generalization to a real customer population or proven churn reduction. See [verification details](docs/verification.md).

## Limits and next steps

- Add real-data evaluation, baseline comparisons, cross-validation and calibration checks.
- Expose and validate explanations before describing the API as explainable.
- Add automated tests and CI; the current repository has no committed test suite.
- Evaluate intervention costs and threshold trade-offs before operational use.
- Experiment tracking, model versioning, drift monitoring, authentication and deployment validation remain planned.

**Skills demonstrated:** Python, Pandas, scikit-learn pipelines, classification, holdout evaluation and FastAPI.
