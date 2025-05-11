import cv2
import seguimiento_manos as sm
import tensorflow as tf
import numpy as np
import time
import mediapipe as mp
from menu import mostrar_menu
import joblib
IMG_WIDTH = 90
IMG_HEIGHT = 60

CATEGORIES = [
    "papel",
    "piedra",
    "tijera",
]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar el escalador
scaler = joblib.load("scaler.pkl")

cap = cv2.VideoCapture(0)
detector = sm.detectorManos(confDeteccion=0.9)

# Variables para rastrear el estado del juego
jugada_realizada = False
tiempo_inicio = 0
cuenta_regresiva = 1  # Tiempo de cuenta regresiva en segundos

def determinarGanador(jugador, ia):
    """
    Determina el ganador del juego de piedra, papel o tijeras.
    
    Parámetros:
    - jugador (int): Elección del jugador (0 = papel, 1 = piedra, 2 = tijera).
    - ia (int): Elección de la IA (0 = papel, 1 = piedra, 2 = tijera).
    
    Retorna:
    - str: Resultado del juego ("Jugador gana", "IA gana", "Empate").
    """
    if jugador == ia:
        return "Empate"
    elif (jugador == 0 and ia == 1) or (jugador == 1 and ia == 2) or (jugador == 2 and ia == 0):
        return "Jugador gana"
    else:
        return "IA gana"

def capturar_puntos(frame):
    """
    Captura los puntos clave de la mano usando MediaPipe y los normaliza.

    Parámetros:
    - frame (numpy.ndarray): Frame capturado por la cámara.

    Retorna:
    - numpy.ndarray: Vector de 63 valores (21 puntos × 3 coordenadas) normalizados o None si no se detecta una mano.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.manos.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            puntos = []
            for landmark in hand_landmarks.landmark:
                puntos.extend([landmark.x, landmark.y, landmark.z])
            
            # Normalizar los puntos clave utilizando el escalador
            puntos = np.array(puntos, dtype=np.float32).reshape(1, -1)
            puntos_normalizados = scaler.transform(puntos)
            return puntos_normalizados.flatten()
    return None


def determinarJugada(puntos, model):
    """
    Utiliza el modelo de IA para predecir si los puntos clave corresponden a piedra, papel o tijera.

    Parámetros:
    - puntos (numpy.ndarray): Vector de 63 valores (21 puntos × 3 coordenadas).
    - model (tf.keras.Model): Modelo de IA entrenado.

    Retorna:
    - str: Predicción ("piedra", "papel" o "tijera").
    """
    if puntos is None or not isinstance(puntos, np.ndarray):
        raise ValueError("Los puntos clave no son válidos. Asegúrate de que sean un numpy.ndarray.")

    # Expandir dimensiones para que sea compatible con el modelo (1, 63)
    puntos_input = np.expand_dims(puntos, axis=0)

    # Realizar la predicción
    prediccion = model.predict(puntos_input, verbose=0)
    print(f"Puntos clave normalizados: {puntos}")
    print(f"Probabilidades: {prediccion}")

    # Obtener la clase con mayor probabilidad
    clase = np.argmax(prediccion)

    # Devolver la categoría correspondiente
    return CATEGORIES[clase]
    

# Mostrar el menú inicial
opcion = mostrar_menu()

if opcion == "salir":
    print("Saliendo del juego...")
    exit()
elif opcion == "info":
    print("Información del juego: Este es un juego de Piedra, Papel o Tijera.")
    opcion = mostrar_menu()  # Volver al menú después de mostrar la información

if opcion == "jugar":
    # Cargar modelo
    modelo = tf.keras.models.load_model("rock_paper_scissors_model.h5")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Crear un fondo más grande (800x600) con un color sólido (negro)
        interfaz = np.zeros((800, 600, 3), dtype=np.uint8)

        # Redimensionar el frame de la cámara para que sea más pequeño (por ejemplo, 480x360)
        frame_resized = cv2.resize(frame, (480, 360))

        # Calcular las coordenadas para centrar el frame en la interfaz
        x_offset = (interfaz.shape[1] - frame_resized.shape[1]) // 2
        y_offset = (interfaz.shape[0] - frame_resized.shape[0]) // 2

        # Colocar el frame de la cámara en el centro de la interfaz
        interfaz[y_offset:y_offset + frame_resized.shape[0], x_offset:x_offset + frame_resized.shape[1]] = frame_resized

        if not jugada_realizada:
            # Mostrar la cuenta regresiva en la interfaz
            tiempo_restante = int(cuenta_regresiva - (time.time() - tiempo_inicio))
            if tiempo_restante > 0:
                cv2.putText(interfaz, f"{tiempo_restante}", (interfaz.shape[1] // 2 - 50, interfaz.shape[0] // 2 - 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            else:
                # Capturar los puntos clave y realizar la predicción
                puntos = capturar_puntos(frame)
                if puntos is not None and puntos.shape == (63,):  # Validar que los puntos no sean None
                    jugada = determinarJugada(puntos, modelo)
                    jugada_realizada = True
                    tiempo_inicio = time.time() + 2  # Esperar 2 segundos antes de reiniciar
                else:
                    print("No se detectó una mano. Intenta nuevamente.")

        if jugada_realizada:
            cv2.putText(interfaz, f"Jugada: {jugada}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if time.time() > tiempo_inicio:
                jugada_realizada = False
                tiempo_inicio = time.time()  # Reiniciar la cuenta regresiva

        # Mostrar la interfaz completa
        cv2.imshow("Piedra, Papel o Tijera", interfaz)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
            break



cap.release()
cv2.destroyAllWindows()