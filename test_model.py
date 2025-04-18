import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

IMG_WIDTH = 90
IMG_HEIGHT = 60

CATEGORIES = [
    "papel",
    "piedra",
    "tijera",
]

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model("rock_paper_scissors_model.h5")

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

def predecirJugada(frame, model):
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
    return CATEGORIES[clase]

def main():
    cap = cv2.VideoCapture(0)
    frame_counter = 0  # Contador para nombrar los archivos de los frames guardados

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Voltear el frame horizontalmente para una vista espejo

        # Mostrar el frame en tiempo real
        cv2.putText(frame, "Presiona 'p' para predecir y guardar el frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Test del modelo", frame)

        # Esperar a que se presione una tecla
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):  # Si se presiona la tecla 'p'
            # Preprocesar el frame
            frame_preprocesado = preprocesar_frame(frame)

            # Guardar el frame preprocesado
            frame_filename = f"frame_{frame_counter}.png"
            cv2.imwrite(frame_filename, frame_preprocesado)
            print(f"Frame guardado como: {frame_filename}")
            frame_counter += 1

            # Realizar la predicción
            prediccion = predecirJugada(frame_preprocesado, modelo)
            print(f"Prediccion: {prediccion}")
            cv2.putText(frame, f"Prediccion: {prediccion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Test del modelo", frame)
            cv2.waitKey(2000)  # Mostrar la predicción durante 2 segundos

        elif key == 27:  # Si se presiona la tecla 'Esc'
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()