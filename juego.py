import cv2
import seguimiento_manos as sm
import tensorflow as tf
import numpy as np
import time
import mediapipe as mp
import pygame
import random

IMG_WIDTH = 90
IMG_HEIGHT = 60

CATEGORIES = [
    "papel",
    "piedra",
    "tijera",
]

# Colores
COLOR_FONDO = (153, 204, 255)  # Celeste
COLOR_BOTON = (255, 255, 255)  # Blanco
COLOR_TEXTO = (0, 0, 0)        # Negro
CUADRO_TAMANO = 300

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
detector = sm.detectorManos(confDeteccion=0.9)

# Variables para rastrear el estado del juego
jugada_realizada = False
tiempo_inicio = 0
cuenta_regresiva = 3  # Tiempo de cuenta regresiva en segundos

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

def iniciar_juego(pantalla):
    """
    Función principal para ejecutar el juego de Piedra, Papel o Tijera en la misma ventana de Pygame.
    """
    global cap, modelo, jugada_realizada, tiempo_inicio  # Asegurarse de que las variables globales sean accesibles

    # Cargar modelo
    modelo = tf.keras.models.load_model("rock_paper_scissors_model.h5")

    ancho, alto = pantalla.get_size()

    # Cargar la imagen grande y dividirla en tres partes
    imagen_grande = pygame.image.load("piedra_papel_tijera.png")  # Asegúrate de usar el nombre correcto
    ancho_imagen = 1856  # Ancho de cada imagen individual
    alto_imagen = 1801   # Alto de cada imagen individual

    tijera = imagen_grande.subsurface((0, 0, ancho_imagen, alto_imagen))
    piedra = imagen_grande.subsurface((ancho_imagen, 0, ancho_imagen, alto_imagen))
    papel = imagen_grande.subsurface((ancho_imagen * 2, 0, ancho_imagen, alto_imagen))

    imagenes_bot = {
        0: papel,  # Imagen para papel
        1: piedra,  # Imagen para piedra
        2: tijera,  # Imagen para tijera
    }

    # Escalar las imágenes del bot para que encajen en el cuadro
    for key in imagenes_bot:
        imagenes_bot[key] = pygame.transform.scale(imagenes_bot[key], (CUADRO_TAMANO, CUADRO_TAMANO))


    # Variables para la jugada del bot
    jugada_bot = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "salir"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "menu"

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Convertir el frame de OpenCV a una superficie de Pygame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        frame_surface = pygame.transform.scale(frame_surface, (CUADRO_TAMANO, CUADRO_TAMANO))  # Escalar la cámara

        # Dibujar el fondo
        pantalla.fill(COLOR_FONDO)

        # Dibujar el cuadro del bot (izquierda)
        pygame.draw.rect(pantalla, COLOR_BOTON, (50, alto // 2 - CUADRO_TAMANO // 2, CUADRO_TAMANO, CUADRO_TAMANO))  # Cuadro del bot
        if jugada_bot is not None:
            pantalla.blit(imagenes_bot[jugada_bot], (50, alto // 2 - CUADRO_TAMANO // 2))  # Imagen del bot

        # Dibujar el cuadro del jugador (derecha)
        pygame.draw.rect(pantalla, COLOR_BOTON, (ancho - CUADRO_TAMANO - 50, alto // 2 - CUADRO_TAMANO // 2, CUADRO_TAMANO, CUADRO_TAMANO))  # Cuadro del jugador
        pantalla.blit(frame_surface, (ancho - CUADRO_TAMANO - 50, alto // 2 - CUADRO_TAMANO // 2))  # Cámara del jugador

        if not jugada_realizada:
            # Mostrar la cuenta regresiva en la ventana de Pygame
            tiempo_restante = int(cuenta_regresiva - (time.time() - tiempo_inicio))
            if tiempo_restante > 0:
                fuente = pygame.font.Font(None, 100)
                texto = fuente.render(str(tiempo_restante), True, COLOR_TEXTO)
                pantalla.blit(texto, (ancho // 2 - texto.get_width() // 2, alto // 2 - texto.get_height() // 2))
            else:
                # Capturar el frame y realizar la predicción
                frame_procesado = preprocesar_frame(frame)
                jugada = determinarJugada(frame_procesado, modelo)

                # Generar la jugada del bot
                jugada_bot = random.randint(0, 2)  # 0 = papel, 1 = piedra, 2 = tijera

                jugada_realizada = True
                tiempo_inicio = time.time() + 2  # Esperar 2 segundos antes de reiniciar

        if jugada_realizada:
            # Mostrar la jugada del jugador
            fuente = pygame.font.Font(None, 50)
            texto_jugador = fuente.render(f"Jugador: {CATEGORIES[jugada]}", True, COLOR_TEXTO)
            pantalla.blit(texto_jugador, (ancho - CUADRO_TAMANO - 50, alto // 2 + CUADRO_TAMANO // 2 + 10))

            # Mostrar la jugada del bot
            texto_bot = fuente.render(f"Bot: {CATEGORIES[jugada_bot]}", True, COLOR_TEXTO)
            pantalla.blit(texto_bot, (50, alto // 2 + CUADRO_TAMANO // 2 + 10))

            # Determinar el ganador
            resultado = determinarGanador(jugada, jugada_bot)
            texto_resultado = fuente.render(f"Resultado: {resultado}", True, COLOR_TEXTO)
            pantalla.blit(texto_resultado, (ancho // 2 - texto_resultado.get_width() // 2, alto - 50))

            if time.time() > tiempo_inicio:
                jugada_realizada = False
                tiempo_inicio = time.time()  # Reiniciar la cuenta regresiva

        pygame.display.flip()