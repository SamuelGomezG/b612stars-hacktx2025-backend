from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List
import pandas as pd
from prophet import Prophet
import io

# Initialize FastAPI app
app = FastAPI(
    title="Bank Balance Forecast API",
    description="Backend wrapper for Prophet model using JSON data input",
    version="1.0"
)

# ---------- Pydantic Models ----------

class BalancePoint(BaseModel):
    date: str  # date as string (ISO format)
    description: str # balance at that date
    amount: float  # balance amount
    balance : float  # balance amount

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        # Validate date format
        try:
            pd.to_datetime(v)
        except Exception:
            raise ValueError(f"Invalid date format for ds: {v}")
        return v

class ForecastRequest(BaseModel):
    data: str
    periods: int = 30  # default 30 future days

# ---------- Helper Function ----------

def run_prophet_model(df: pd.DataFrame, periods: int = 30):
    # ✅ Your exact Prophet model configuration
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='paycheck', period=14, fourier_order=5)
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    return result

# ---------- API Routes ----------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast")
def forecast(request: ForecastRequest):

    csv_data = request.data
    buffer = io.StringIO(csv_data)
    df = pd.read_csv(buffer)

    prediction_df = pd.DataFrame({
        'ds': pd.to_datetime(df['date']),
        'y': df['balance']
    })

    try:
        forecast_result = run_prophet_model(prediction_df, periods=request.periods)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    return forecast_result.to_dict(orient="records")
