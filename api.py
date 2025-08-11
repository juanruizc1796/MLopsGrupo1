from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Montar carpeta 'static' para archivos estáticos (donde va el index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Endpoint raíz para servir index.html
@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# Cargar modelos al inicio
models = {
    "rf": joblib.load("models/rf_model.pkl"),
    "knn": joblib.load("models/knn_model.pkl")
}

# Diccionario para convertir predicciones a nombre especie
species_mapping = {
    0: "Adelie",
    1: "Chinstrap",
    2: "Gentoo"
}

# Esquema entrada
class PenguinInput(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float

# Endpoint predicción
@app.post("/predict")
def predict(data: PenguinInput, model_name: str = Query("rf", enum=["rf", "knn"])):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Modelo no disponible")
    model = models[model_name]
    features = np.array([[data.bill_length_mm, data.bill_depth_mm, data.flipper_length_mm, data.body_mass_g]])
    prediction = model.predict(features)[0]
    return {
        "prediction": int(prediction),
        "especie": species_mapping[int(prediction)],
        "modelo_usado": model_name
    }