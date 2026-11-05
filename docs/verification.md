# Verification record

Verified locally on 2026-11-05 using Python 3.13.7 on macOS ARM64.
Source commit: `8564853248078b31880c36d14eb1f6617cb4abfd`. This documentation change does not alter model or API code.

## Reproduce

Follow the README setup from the repository root. [verified-environment.txt](verified-environment.txt) records all installed package versions, including HTTPX used for verification. It is an environment snapshot, not a cross-platform lockfile. Direct dependencies are pinned in requirements.txt; transitive dependencies otherwise resolve at installation time.

Command executed:
```bash
python src/train.py
```

Observed:
```text
ROC-AUC: 0.772
              precision    recall  f1-score   support
stay               0.83      0.71      0.76       202
churn              0.54      0.70      0.61        98
accuracy                               0.71       300
macro avg          0.69      0.71      0.69       300
weighted avg       0.74      0.71      0.71       300
```

The environment emitted a SciPy OptimizeWarning about the unknown solver option `iprint`; training completed and inference succeeded. Dependency compatibility deserves a follow-up before deployment. No committed automated tests were present.

## API checks performed

Using FastAPI TestClient and HTTPX 0.28.1 against the trained model:
- GET /health returned ready.
- The README example returned HTTP 200 and the displayed JSON response.
- A negative tenure_months returned HTTP 422.
- With the model path temporarily pointed to an absent file, health returned model_not_trained and inference returned HTTP 503.

These were local smoke checks, not a full test suite or a deployed-network check.

## Not verified

Docker build/run, Windows setup, Python 3.12 execution, hosted deployment, real-data performance, business impact, probability calibration and explanation quality.
No screenshots are supplied because no finished dashboard or deployed application was verified.
