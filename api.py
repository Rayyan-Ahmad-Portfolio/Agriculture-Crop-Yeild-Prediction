import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="AgriYield Cloud MLOps API")

STATIC_DIR = Path(__file__).parent / "static"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model and preprocessor safely
try:
    with open('dtr.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except Exception as e:
    print(f"Error loading pickle objects: {e}")

class CropInput(BaseModel):
    Area: str
    Item: str
    Year: int
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float

@app.post("/predict")
def predict_yield(data: CropInput):
    try:
        # Reconstruct into a DataFrame to preserve feature names for the encoder/scaler
        df_input = pd.DataFrame([{
            'Area': data.Area,
            'Item': data.Item,
            'Year': data.Year,
            'average_rain_fall_mm_per_year': data.average_rain_fall_mm_per_year,
            'pesticides_tonnes': data.pesticides_tonnes,
            'avg_temp': data.avg_temp
        }])
        
        # Transform data using your trained column transformer pipeline
        transformed = preprocessor.transform(df_input)
        prediction = model.predict(transformed)[0]
        
        return {
            "status": "success",
            "hg_ha_yield": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def serve_frontend():
    """Serve the HTML harvest console at the API root."""
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
