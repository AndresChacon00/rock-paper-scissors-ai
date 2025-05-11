import cv2
import seguimiento_manos as sm
import tensorflow as tf
import numpy as np
import time
import mediapipe as mp

IMG_WIDTH = 90
IMG_HEIGHT = 60

CATEGORIES = [
    "papel",
    "piedra",
    "tijera",
]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
detector = sm.detectorManos(confDeteccion=0.9)

# Variables para rastrear el estado del juego
jugada_realizada = False
tiempo_inicio = 0
cuenta_regresiva = 3  # Tiempo de cuenta regresiva en segundos

def iniciar_juego(pantalla):
    """
    Ejecuta el juego de Piedra, Papel o Tijera.
    """
    modelo = tf.keras.models.load_model("rock_paper_scissors_model.h5")
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    detector = sm.detectorManos(confDeteccion=0.9)

    # Variables para rastrear el estado del juego
    jugada_realizada = False
    tiempo_inicio = 0
    cuenta_regresiva = 3  # Tiempo de cuenta regresiva en segundos

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
                # Capturar el frame y realizar la predicción
                frame_procesado = preprocesar_frame(frame)
                cv2.imwrite("frame.png", frame_procesado)
                jugada = determinarJugada(frame_procesado, modelo)
                jugada_realizada = True
                tiempo_inicio = time.time() + 2  # Esperar 2 segundos antes de reiniciar

        if jugada_realizada:
            cv2.putText(interfaz, f"Jugada: {CATEGORIES[jugada]}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if time.time() > tiempo_inicio:
                jugada_realizada = False
                tiempo_inicio = time.time()  # Reiniciar la cuenta regresiva

        # Mostrar la interfaz completa
        cv2.imshow("Piedra, Papel o Tijera", interfaz)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
            break

    cap.release()
    cv2.destroyAllWindows()

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

def preprocesar_frame(frame):
    """
    Preprocesa el frame para detectar la mano y generar una imagen en blanco y negro, manteniendo la calidad.

    Parámetros:
    - frame (numpy.ndarray): Frame capturado por la cámara.

    Retorna:
    - numpy.ndarray: Frame preprocesado con la mano en blanco y negro, redimensionado al tamaño esperado por el modelo.
    """
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Convertir el frame a RGB (MediaPipe requiere RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame con MediaPipe
        results = hands.process(frame_rgb)

        # Crear una máscara negra del mismo tamaño que el frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Crear una lista de puntos (x, y) a partir de los landmarks
                points = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    points.append((x, y))
                
                # Extender la región hacia abajo para incluir más de la muñeca
                wrist_extension = 50  # Ajusta este valor para incluir más o menos de la muñeca
                points.append((points[0][0], points[0][1] + wrist_extension))  # Extender hacia abajo

                # Convertir los puntos a un array de NumPy
                points = np.array(points, dtype=np.int32)

                # Dibujar un polígono que conecte todos los puntos de referencia
                cv2.fillPoly(mask, [points], 255)  # Rellenar el área de la mano con blanco

            # Dilatar la máscara para incluir más área alrededor de la mano
            kernel = np.ones((30, 30), np.uint8)  # Ajusta el tamaño del kernel según sea necesario
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Aplicar la máscara al frame original
        hand_region = cv2.bitwise_and(frame, frame, mask=mask)

        # Redimensionar al tamaño esperado por el modelo
        resized = cv2.resize(hand_region, (IMG_WIDTH, IMG_HEIGHT))

        return resized

def determinarJugada(frame, model):
    """
    Utiliza el modelo de IA para predecir si el frame muestra piedra, papel o tijera.

    Parámetros:
    - frame (numpy.ndarray): Frame capturado por la cámara.
    - model (tf.keras.Model): Modelo de IA entrenado.

    Retorna:
    - str: Predicción ("piedra", "papel" o "tijera").
    """
    # Preprocesar el frame
    frame_preprocesado = preprocesar_frame(frame)

    # Normalizar los valores de píxeles (0-255 -> 0-1)
    frame_normalizado = frame_preprocesado / 255.0

    # Expandir dimensiones para que sea compatible con el modelo (1, IMG_HEIGHT, IMG_WIDTH, 3)
    frame_input = np.expand_dims(frame_normalizado, axis=0)

    # Realizar la predicción
    prediccion = model.predict(frame_input, verbose=0)
    print(f"Probabilidades: {prediccion}")

    # Obtener la clase con mayor probabilidad
    clase = np.argmax(prediccion)

    # Devolver la categoría correspondiente
    return clase