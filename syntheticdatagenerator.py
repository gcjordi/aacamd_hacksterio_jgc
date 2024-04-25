import pandas as pd
import numpy as np

# Configuración de los parámetros para la generación de datos sintéticos
n_samples = 10000  # Número de muestras a generar

# Definición de las distribuciones y características de los datos
np.random.seed(0)
data = {
    "Edad": np.random.randint(40, 91, n_samples),
    "Sexo": np.random.choice(["Masculino", "Femenino"], n_samples),
    "Historia familiar de Alzheimer": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    "Índice de Placa": np.random.uniform(0, 5, n_samples),
    "Profundidad de Bolsa Periodontal": np.random.uniform(1, 4, n_samples),
    "Número de dientes perdidos": np.random.randint(0, 33, n_samples),
    "Frecuencia de sangrado en encías al cepillar": np.random.choice(["Nunca", "Ocasionalmente", "Frecuentemente", "Siempre"], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
    "Nivel de higiene oral (auto-reportado)": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    "Diabetes": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    "Hipertensión": np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    "Nivel educativo": np.random.choice(["Sin educación", "Educación primaria", "Educación secundaria", "Universitaria"], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
    "Consumo de tabaco": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    "Actividad física": np.random.choice(["Sedentario", "Moderadamente activo", "Muy activo"], n_samples, p=[0.4, 0.4, 0.2]),
    "Dieta": np.random.choice(["Pobre", "Moderada", "Excelente"], n_samples, p=[0.3, 0.6, 0.1]),
    "Score cognitivo inicial": np.random.uniform(70, 100, n_samples),
    "Desarrollo de Alzheimer en 5 años": np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
}

# Creación del DataFrame
df = pd.DataFrame(data)

df.head()
