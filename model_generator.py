import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
# Cargar el dataset
data = pd.read_csv("hand_gesture_dataset_mixto.csv")

# Separar características (X) y etiquetas (y)
X = data.drop("label", axis=1).values  # Todas las columnas excepto "label"
y = data["label"].values  # Columna "label"

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar las características
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Guardar el escalador
joblib.dump(scaler,"scaler.pkl")

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Entrada con el número de características
    tf.keras.layers.Dense(128, activation="relu"),  # Capa oculta con 64 neuronas y activación ReLU
    tf.keras.layers.Dropout(0.3), # Dropout
    tf.keras.layers.Dense(128, activation="relu"),  # Capa oculta con 64 neuronas y activación ReLU
    tf.keras.layers.Dropout(0.3), # Dropout
    tf.keras.layers.Dense(len(set(y)), activation="softmax")  # Capa de salida con activación softmax
])

# Compilar el modelo
model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10)

# Evaluar el modelo
model.evaluate(X_test, y_test, verbose=2)

# Save model
model.save("rock_paper_scissors_model.h5")

# Generar un informe de clasificación
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred_classes))