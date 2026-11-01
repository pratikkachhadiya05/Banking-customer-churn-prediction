# Banking Customer Churn Prediction

A machine learning web application that predicts whether a bank customer is likely to leave (churn). It pairs a trained **Random Forest** pipeline with a **FastAPI** backend and a multi-page dashboard, so bank teams can score customers, see the risk level, and understand what generally drives churn.

## Features

- **Churn prediction:** enter customer details and get a prediction (*At risk* or *Likely to stay*)
- **Probabilities:** churn and retain probability as percentages
- **Risk levels:** Low, Medium or High, based on churn probability
- **Model-level feature importance:** the top factors that generally drive churn in the trained model
- **Multi-page dashboard:** Dashboard, Churn Predictor, Customer Analytics, Model Performance and Retention Campaigns
- **REST API:** validated JSON endpoint (`POST /predict`) with auto-generated docs at `/docs`

## Tech Stack

- **Backend:** Python, FastAPI, Pydantic, Jinja2
- **ML:** scikit-learn (pipeline with `RandomizedSearchCV`), Random Forest, pandas
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates + static assets)

## Project Structure

```
Banking-customer-churn-prediction/
├── Datasets/               # Customer churn dataset
├── static/                 # CSS, JS and images
├── templates/              # HTML pages
│   ├── Dashboard.html
│   ├── Churn_predictor.html
│   ├── Customer_analytics.html
│   ├── Model_performance.html
│   └── Retention_Campaigns.html
├── app.py                  # FastAPI application
├── model.ipynb             # Notebook: data prep, training, evaluation
├── bank_churn_model.pkl    # Trained model
├── requirements.txt        # Python dependencies
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/pratikkachhadiya05/Banking-customer-churn-prediction.git
cd Banking-customer-churn-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** for the dashboard. Interactive API docs are at **http://127.0.0.1:8000/docs**.

## Pages

| Route                   | Page                                   |
|-------------------------|----------------------------------------|
| `/` or `/Dashboard`     | Main dashboard                         |
| `/Churn_predictor`      | Score an individual customer           |
| `/Customer_analytics`   | Customer analytics                     |
| `/Model_performance`    | Model performance metrics              |
| `/Retention_Campaigns`  | Retention campaign planning            |

## Input Fields

| Field             | Type   | Validation                |
|-------------------|--------|---------------------------|
| `Gender`          | string | `Male` or `Female`        |
| `Age`             | int    | 18 – 100                  |
| `Tenure`          | int    | 0 – 15 (years)            |
| `CreditScore`     | int    | 300 – 900                 |
| `Balance`         | float  | 0 or more                 |
| `NumOfProducts`   | int    | 1 – 4                     |
| `HasCrCard`       | int    | 0 or 1                    |
| `IsActiveMember`  | int    | 0 or 1                    |
| `EstimatedSalary` | float  | 0 or more                 |

## API Usage

### `POST /predict`

**Request**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Female",
    "Age": 42,
    "Tenure": 3,
    "CreditScore": 650,
    "Balance": 85000,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 60000
  }'
```

**Response** (values are illustrative)

```json
{
  "prediction": 1,
  "label": "At risk",
  "churn_probability": 62.5,
  "retain_probability": 37.5,
  "risk_level": "Medium",
  "top_factors": [
    { "label": "Age", "importance": 24.1 },
    { "label": "Number of products", "importance": 18.3 }
  ]
}
```

### Risk levels

| Churn probability | Risk level |
|-------------------|------------|
| 66% and above     | High       |
| 33% to under 66%  | Medium     |
| Below 33%         | Low        |

## Model

The full workflow (data exploration, preprocessing, model training with hyperparameter search, and evaluation) is in [`model.ipynb`](model.ipynb). The trained pipeline is saved as `bank_churn_model.pkl` and loaded by `app.py` at startup.

> **Note:** the feature importances shown with each prediction are **global** (model-level) values, not per-customer explanations. Per-customer explanations would need a tool such as SHAP or LIME.

## Limitations

- Trained on a historical dataset; accuracy on real bank data is not guaranteed.
- Intended for learning and demonstration, not for real lending or retention decisions without further validation.

## Author

**Pratik Kachhadiya**: [@pratikkachhadiya05](https://github.com/pratikkachhadiya05)
