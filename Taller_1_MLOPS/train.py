import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Crear carpeta de modelos si no existe
os.makedirs("models", exist_ok=True)

# Cargar datos
df = load_penguins()
df.dropna(inplace=True)

# Features y target
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = LabelEncoder().fit_transform(df["species"])

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar RandomForest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "models/rf_model.pkl")

# Entrenar KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
joblib.dump(knn, "models/knn_model.pkl")

print("Modelos entrenados y guardados en carpeta 'models'")
