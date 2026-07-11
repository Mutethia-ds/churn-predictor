# API Reference

The FastAPI app lives at `src/churn/api/main.py`. Once running (`make api`),
interactive docs are auto-generated at `http://localhost:8000/docs`.

## `GET /health`

Returns service status and whether a trained model is loaded.

```json
{"status": "ok", "model_loaded": true}
```

## `POST /predict`

Request body:

```json
{
  "tenure_months": 12,
  "contract_type": "month-to-month",
  "internet_service": "fiber",
  "payment_method": "electronic_check",
  "monthly_charges": 74.5,
  "total_charges": 894.0,
  "support_calls": 2
}
```

Response:

```json
{"churn_probability": 0.63, "churn_prediction": 1}
```
