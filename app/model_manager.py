import os
from typing import Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
import dill

class ModelManager:
    def __init__(self, model_paths: Dict[str, str]):
        """
        model_paths: diccionario con el orden de carga de los modelos
        Ej:
        {
            "text_pipeline": "models/text_pipeline.pkl",
            "scaler_x": "models/scaler_x.pkl",
            "best_model": "models/best_model_multiinput.keras"
        }
        """
        self.model_paths = model_paths
        self.models: Dict[str, Any] = {}
        self.model_emb = None

    def load_model_if_needed(self, name: str):
        if name in self.models:
            return self.models[name]

        path = self.model_paths.get(name)
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Archivo {path} no existe")

        # Carga Keras o pickle/dill
        if path.endswith(".keras") or path.endswith(".h5"):
            self.models[name] = load_model(path)
        else:
            with open(path, "rb") as f:
                self.models[name] = dill.load(f)

        print(f"[INFO] Modelo {name} cargado correctamente.")
        return self.models[name]

    def scale_numeric(self, X_numeric):
        scaler = self.load_model_if_needed('scaler_x')
        return scaler.transform(X_numeric)
    
    def scale_numeric_y(self, Y_numeric):
        scaler_y = self.load_model_if_needed('scaler_y')
        return scaler_y.inverse_transform(Y_numeric)

    def predict_final(self, X_emb, X_scaled):
        model = self.load_model_if_needed('best_model')
        return model.predict([X_emb, X_scaled])
    
    def get_embedding_model(self):
        if self.model_emb is None:
            self.model_emb = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model_emb