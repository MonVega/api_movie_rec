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
        self.models: Dict[str, Any] = {}
        self.model_emb = None 
        self.load_models(model_paths)

    def load_models(self, model_paths: Dict[str, str]):
        for name, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Archivo {path} no existe")
            
            # Carga Keras o pickle/dill
            if path.endswith(".keras") or path.endswith(".h5"):
                self.models[name] = load_model(path)
            else:
                with open(path, "rb") as f:
                    self.models[name] = dill.load(f)
            
            print(f"[INFO] Modelo {name} cargado correctamente.")

    def scale_numeric(self, X_numeric):
        """
        Aplica scaler_x a la parte numeric
        """
        if 'scaler_x' not in self.models:
            raise ValueError("El modelo 'scaler_x' no está cargado")
        return self.models['scaler_x'].transform(X_numeric)
    
    def scale_numeric_y(self, Y_numeric):
        """
        Aplica scaler_x a la parte numeric
        """
        if 'scaler_y' not in self.models:
            raise ValueError("El modelo 'scaler_x' no está cargado")
        return self.models['scaler_y'].inverse_transform(Y_numeric)


    def predict_final(self, X_emb, X_scaled):
        """
        Predicción con best_model_multiinput.keras usando 2 inputs
        """
        if 'best_model' not in self.models:
            raise ValueError("El modelo 'best_model' no está cargado")
        
        model = self.models['best_model']
        return model.predict([X_emb, X_scaled])
    
    def get_embedding_model(self):
        if self.model_emb is None:
            self.model_emb = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model_emb