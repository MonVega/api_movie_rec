from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .model_manager import ModelManager
from .text_processing import process_text, split_text_numeric
import pandas as pd

# Solo guardamos rutas de modelos, no los cargamos aún
model_paths = {
    "scaler_x": "models/scaler_x.pkl",
    "scaler_y": "models/scaler_y.pkl",
    "best_model": "models/best_model_mulinput.keras"
}

manager = ModelManager(model_paths)  # Lazy loading
app = FastAPI(title="Movie Rating Prediction API")

# Configuración CORS para aceptar POST desde cualquier dominio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"message": "Bienvenido a la API de predicción de ratings 🎬"}

@app.post("/predict")
async def predict(
    col1: str = Form(...),
    col2: str = Form(...)
):
    """
    Endpoint que recibe 2 strings desde un formulario
    y hace todo el flujo de predicción usando ModelManager.
    """
    try:
        # Convertir a DataFrame
        df = pd.DataFrame({'title': [col1], 'description': [col2]})
        
        # Preprocesamiento
        df_process = process_text(df)
        df_split = split_text_numeric(df_process)
        X_numeric = df_split['numeric']
        X_emb = manager.get_embedding_model().encode(df_split['text'])

        # Escalar numeric (lazy loading)
        X_scaled = manager.scale_numeric(X_numeric)

        # Predicción final con modelo multi-input (lazy loading)
        prediction = manager.predict_final(X_emb, X_scaled)

        # Escalar predicción
        Y_scaled = manager.scale_numeric_y(prediction)

        # Convertir predicción a float
        prediction_val = float(Y_scaled[0][0])

        return JSONResponse(content={"prediction": prediction_val})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})