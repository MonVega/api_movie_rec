from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .model_manager import ModelManager
from .text_processing import process_text,split_text_numeric
import pandas as pd

model_paths = {
    "scaler_x": "models/scaler_x.pkl",
    "scaler_y":"models/scaler_y.pkl",
    "best_model": "models/best_model_mulinput.keras"
}

manager = ModelManager(model_paths)
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
        
        df_process = process_text(df)
        df_split = split_text_numeric(df_process)
        X_numeric = df_split['numeric']
        X_emb = manager.get_embedding_model().encode(df_split['text'])

        # 2. Escalar numeric
        X_scaled = manager.scale_numeric(X_numeric)

        # 3. Predicción final con modelo multi-input
        prediction = manager.predict_final(X_emb, X_scaled)
        print(prediction)
        Y_scaled = manager.scale_numeric_y(prediction)
        # Convertir predicción a lista o float
        prediction_val = prediction.tolist() if hasattr(prediction, "tolist") else float(prediction)

        return JSONResponse(content={"prediction": float(Y_scaled[0][0])})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})