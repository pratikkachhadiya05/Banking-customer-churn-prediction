import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

app = FastAPI(title="Customer Churn Prediction API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Load model -------------------------------------------------------

MODEL_PATH = "bank_churn_model.pkl"

try:
    with open(MODEL_PATH, "rb") as model_file:
        churn_model = pickle.load(model_file)
    # RandomizedSearchCV wraps the fitted pipeline; unwrap once here.
    fitted_pipeline = getattr(churn_model, "best_estimator_", churn_model)
    print("Model loaded successfully.")
except FileNotFoundError:
    churn_model = None
    fitted_pipeline = None
    print(f"Error: model file not found at {MODEL_PATH}.")

# Global feature importances (from the trained RandomForest), used to show
# "what generally drives churn" context next to a prediction. These are NOT
# per-customer explanations (that would require SHAP/LIME), just model-level
# reference info, and the UI is careful to label them as such.
FEATURE_LABELS = {
    "ohe__Gender_Male": "Gender",
    "sc__CreditScore": "Credit score",
    "sc__Age": "Age",
    "sc__Tenure": "Tenure (years)",
    "sc__Balance": "Account balance",
    "sc__NumOfProducts": "Number of products",
    "sc__HasCrCard": "Has credit card",
    "sc__IsActiveMember": "Active membership",
    "sc__EstimatedSalary": "Estimated salary",
}


def get_global_feature_importance():
    if fitted_pipeline is None:
        return []
    try:
        transformer = fitted_pipeline.named_steps["transformer"]
        classifier = fitted_pipeline.named_steps["classifier"]
        names = transformer.get_feature_names_out()
        importances = classifier.feature_importances_
        pairs = sorted(zip(names, importances), key=lambda p: p[1], reverse=True)
        return [
            {"label": FEATURE_LABELS.get(n, n), "importance": round(float(v) * 100, 1)}
            for n, v in pairs
        ]
    except Exception:
        return []


# --- Request/response schemas ------------------------------------------

class CustomerData(BaseModel):
    Gender: str = Field(..., pattern="^(Male|Female)$")
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=15)
    CreditScore: int = Field(..., ge=300, le=900)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    churn_probability: float
    retain_probability: float
    risk_level: str
    top_factors: list


# --- Page routes ---------------------------------------------------------

PAGES = {
    "": "Dashboard.html",
    "Dashboard": "Dashboard.html",
    "Churn_predictor": "Churn_predictor.html",
    "Customer_analytics": "Customer_analytics.html",
    "Model_performance": "Model_performance.html",
    "Retention_Campaigns": "Retention_Campaigns.html",
}


def render_page(request: Request, template_name: str):
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context={"title": "Customer Churn Prediction Dashboard"},
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render_page(request, "Dashboard.html")


@app.get("/Dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return render_page(request, "Dashboard.html")


@app.get("/Churn_predictor", response_class=HTMLResponse)
async def churn_predictor_page(request: Request):
    return render_page(request, "Churn_predictor.html")


@app.get("/Customer_analytics", response_class=HTMLResponse)
async def customer_analytics(request: Request):
    return render_page(request, "Customer_analytics.html")


@app.get("/Model_performance", response_class=HTMLResponse)
async def model_performance(request: Request):
    return render_page(request, "Model_performance.html")


@app.get("/Retention_Campaigns", response_class=HTMLResponse)
async def retention_campaigns(request: Request):
    return render_page(request, "Retention_Campaigns.html")


# --- Prediction API --------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    if fitted_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

    row = pd.DataFrame([{
        "CreditScore": customer.CreditScore,
        "Gender": customer.Gender,
        "Age": customer.Age,
        "Tenure": customer.Tenure,
        "Balance": customer.Balance,
        "NumOfProducts": customer.NumOfProducts,
        "HasCrCard": customer.HasCrCard,
        "IsActiveMember": customer.IsActiveMember,
        "EstimatedSalary": customer.EstimatedSalary,
    }])

    try:
        pred = int(fitted_pipeline.predict(row)[0])
        proba = fitted_pipeline.predict_proba(row)[0]
        churn_prob = float(proba[1])
        retain_prob = float(proba[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    if churn_prob >= 0.66:
        risk_level = "High"
    elif churn_prob >= 0.33:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    print(f"Prediction: {pred}, Churn probability: {churn_prob:.2f}, Retain probability: {retain_prob:.2f}, Risk level: {risk_level}")
    return PredictionResponse(
        prediction=pred,
        label="At risk" if pred == 1 else "Likely to stay",
        churn_probability=round(churn_prob * 100, 1),
        retain_probability=round(retain_prob * 100, 1),
        risk_level=risk_level,
        top_factors=get_global_feature_importance()[:5],
    )