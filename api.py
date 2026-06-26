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
model = None
preprocessor = None
model_load_error = None

try:
    with open('dtr.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except Exception as e:
    model_load_error = str(e)
    print(f"Error loading pickle objects: {e}")

class CropInput(BaseModel):
    Area: str
    Item: str
    Year: int
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float

@app.get("/health")
def health_check():
    """Lightweight status check for the HTML dashboard."""
    return {
        "status": "ok",
        "model_loaded": model is not None and preprocessor is not None,
        "error": model_load_error,
    }

@app.post("/predict")
def predict_yield(data: CropInput):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail=model_load_error or "Model files (dtr.pkl, preprocessor.pkl) are not loaded.",
        )
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
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/styles.css", include_in_schema=False)
def serve_styles():
    """Serve CSS when the page is loaded from /."""
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def serve_app_js():
    """Serve JS when the page is loaded from /."""
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
