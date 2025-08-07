from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np

# Crear la app FastAPI
app = FastAPI()

# Cargar modelos al inicio
models = {
    "rf": joblib.load("models/rf_model.pkl"),
    "knn": joblib.load("models/knn_model.pkl")
}

# Diccionario para convertir predicciones a nombres de especies
species_mapping = {
    0: "Adelie",
    1: "Chinstrap",
    2: "Gentoo"
}

# Esquema de entrada esperado
class PenguinInput(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float

# Endpoint de prueba
@app.get("/")
def home():
    return {"message": "API para predecir especie de pingüinos"}

# Endpoint de predicción
@app.post("/predict")
def predict(data: PenguinInput, model_name: str = Query("rf", enum=["rf", "knn"])):
    # Validar si el modelo está disponible
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Modelo no disponible")

    # Seleccionar modelo
    model = models[model_name]

    # Convertir entrada a arreglo
    features = np.array([[data.bill_length_mm, data.bill_depth_mm, data.flipper_length_mm, data.body_mass_g]])

    # Hacer la predicción
    prediction = model.predict(features)[0]

    # Devolver respuesta
    return {
        "prediction": int(prediction),
        "especie": species_mapping[int(prediction)],
        "modelo_usado": model_name
    }
