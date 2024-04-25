# Modelo de Predicción de Alzheimer basado en la Salud de las Encías
#Este notebook guía a través del proceso de entrenamiento de un modelo de machine learning para predecir si una persona podría desarrollar Alzheimer en los próximos 5 años, basándose en datos sobre la salud de sus encías.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargamos los datos desde el archivo CSV
data = pd.read_csv('Dataset_Salud_Encias_Alzheimer.csv')
data.head()

# Codificación de variables categóricas
label_encoders = {}
for column in ['Sexo', 'Frecuencia de sangrado en encías al cepillar', 'Nivel educativo', 'Actividad física', 'Dieta']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# División del dataset en conjunto de entrenamiento y prueba
X = data.drop('Desarrollo de Alzheimer en 5 años', axis=1)
y = data['Desarrollo de Alzheimer en 5 años']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar un modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predicciones del modelo
y_pred = model.predict(X_test_scaled)

# Evaluación del modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

