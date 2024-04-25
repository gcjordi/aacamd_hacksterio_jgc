from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Codificación de variables categóricas
label_encoders = {}
categorical_vars = ['Sexo', 'Frecuencia de sangrado en encías al cepillar', 'Nivel educativo', 'Actividad física', 'Dieta']
for column in categorical_vars:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# División del dataset en conjunto de entrenamiento y prueba
X = df.drop('Desarrollo de Alzheimer en 5 años', axis=1)
y = df['Desarrollo de Alzheimer en 5 años']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predicciones del modelo y evaluación
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Importancia de las características
feature_importances = model.feature_importances_
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

accuracy, features_df.head()
