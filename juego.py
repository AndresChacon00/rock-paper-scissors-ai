import cv2
import seguimiento_manos as sm
import tensorflow as tf
import numpy as np
import time
import mediapipe as mp
import pygame
import random
from menu import lluvia_imagenes
import joblib
import sys
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
CUADRO_TAMANO = 500

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar el escalador
scaler = joblib.load("scaler.pkl")

cap = cv2.VideoCapture(0)
detector = sm.detectorManos(confDeteccion=0.9)

# Variables para rastrear el estado del juego
jugada_realizada = False

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
    return clase

def seleccionar_modo(pantalla):
    """
    Muestra una pantalla para seleccionar el modo de juego.
    """
    ancho, alto = pantalla.get_size()
    fuente = pygame.font.Font(None, 50)
    titulo = pygame.font.Font(None, 80).render("Selecciona el modo de juego", True, COLOR_TEXTO)

    # Botones para los modos
    botones = {
        "2 de 3": pygame.Rect((ancho - 300) // 2, alto // 2 - 60, 300, 50),
        "3 de 5": pygame.Rect((ancho - 300) // 2, alto // 2 + 20, 300, 50),
    }

    # Cargar imágenes para la lluvia
    imagen_grande = pygame.image.load("piedra_papel_tijera.png")
    ancho_imagen = 1856
    alto_imagen = 1801

    tijera = pygame.transform.scale(imagen_grande.subsurface((0, 0, ancho_imagen, alto_imagen)), (100, 100))
    piedra = pygame.transform.scale(imagen_grande.subsurface((ancho_imagen, 0, ancho_imagen, alto_imagen)), (100, 100))
    papel = pygame.transform.scale(imagen_grande.subsurface((ancho_imagen * 2, 0, ancho_imagen, alto_imagen)), (100, 100))

    imagenes = [piedra, papel, tijera]
    lluvia = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "salir"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if botones["2 de 3"].collidepoint(event.pos):
                    return 3  # Gana el primero en llegar a 2 de 3
                elif botones["3 de 5"].collidepoint(event.pos):
                    return 5  # Gana el primero en llegar a 3 de 5

        pantalla.fill(COLOR_FONDO)

        # Llamar a la función de lluvia
        lluvia_imagenes(pantalla, lluvia, imagenes, ancho, alto)

        pantalla.blit(titulo, (ancho // 2 - titulo.get_width() // 2, alto // 4))

        for texto, rect in botones.items():
            borde_rect = rect.inflate(4, 4)  # Crear un borde ligeramente más grande
            pygame.draw.rect(pantalla, (0, 0, 0), borde_rect)  # Dibujar el borde negro
            pygame.draw.rect(pantalla, COLOR_BOTON, rect)  # Dibujar el botón blanco
            texto_render = fuente.render(texto, True, COLOR_TEXTO)
            pantalla.blit(texto_render, (rect.x + (rect.width - texto_render.get_width()) // 2, rect.y + 10))

        pygame.display.flip()

def mostrar_pantalla_final(pantalla, mensaje):
    """
    Muestra una pantalla final con un mensaje y espera unos segundos antes de regresar al menú.
    
    Parámetros:
    - pantalla: La superficie de Pygame donde se dibujará el mensaje.
    - mensaje: El mensaje a mostrar (por ejemplo, "¡Ganaste!" o "Perdiste").
    """
    ancho, alto = pantalla.get_size()
    fuente = pygame.font.Font(None, 100)
    tiempo_espera = 3  # Segundos antes de regresar al menú

    # Cargar imágenes para la lluvia
    imagen_grande = pygame.image.load("piedra_papel_tijera.png")
    ancho_imagen = 1856
    alto_imagen = 1801

    tijera = pygame.transform.scale(imagen_grande.subsurface((0, 0, ancho_imagen, alto_imagen)), (100, 100))
    piedra = pygame.transform.scale(imagen_grande.subsurface((ancho_imagen, 0, ancho_imagen, alto_imagen)), (100, 100))
    papel = pygame.transform.scale(imagen_grande.subsurface((ancho_imagen * 2, 0, ancho_imagen, alto_imagen)), (100, 100))

    imagenes = [piedra, papel, tijera]
    lluvia = []

    tiempo_inicio = time.time()
    while time.time() - tiempo_inicio < tiempo_espera:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pantalla.fill(COLOR_FONDO)

        # Llamar a la función de lluvia
        lluvia_imagenes(pantalla, lluvia, imagenes, ancho, alto)

        texto = fuente.render(mensaje, True, COLOR_TEXTO)
        pantalla.blit(texto, (ancho // 2 - texto.get_width() // 2, alto // 2 - texto.get_height() // 2))
        pygame.display.flip()

def iniciar_juego(pantalla):
    """
    Función principal para ejecutar el juego de Piedra, Papel o Tijera en la misma ventana de Pygame.
    """
    global cap, modelo, jugada_realizada, tiempo_inicio  # Asegurarse de que las variables globales sean accesibles

    # Seleccionar el modo de juego
    modo = seleccionar_modo(pantalla)
    if modo == "salir":
        return "salir"
    
    # Variables para el puntaje
    puntaje_jugador = 0
    puntaje_bot = 0

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
    jugada = None

    # Variable para el botón de inicio
    juego_iniciado = False
    tiempo_inicio = 0
    cuenta_regresiva = 5  # Contador inicial de 5 segundos

    # Inicializar el botón "Empezar"
    fuente = pygame.font.Font(None, 80)
    texto_empezar = fuente.render("Empezar", True, COLOR_TEXTO)
    boton_empezar = texto_empezar.get_rect(center=(ancho // 2, alto // 2))

    # Variable para controlar si es la primera ronda
    primera_ronda = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "salir"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "menu"
            if not juego_iniciado and event.type == pygame.MOUSEBUTTONDOWN:
                # Detectar clic en el botón "Empezar"
                if boton_empezar.collidepoint(event.pos):
                    juego_iniciado = True
                    tiempo_inicio = time.time()  # Iniciar el contador de 5 segundos

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
        if not juego_iniciado:
            # Dibujar un rectángulo semitransparente sobre el fondo
            overlay = pygame.Surface((ancho, alto))  # Crear una superficie del tamaño de la pantalla
            overlay.set_alpha(128)  # Establecer la transparencia (0 = completamente transparente, 255 = opaco)
            overlay.fill((0, 0, 0))  # Color negro semitransparente
            pantalla.blit(overlay, (0, 0))  # Dibujar la superficie sobre la pantalla

            # Dibujar el borde negro del botón
            borde_boton = boton_empezar.inflate(4, 4)  # Ajustar el tamaño del borde
            pygame.draw.rect(pantalla, (0, 0, 0), borde_boton)  # Dibujar el borde negro
            # Dibujar el botón blanco dentro del borde
            pygame.draw.rect(pantalla, COLOR_BOTON, boton_empezar.inflate(16, 16))
            pantalla.blit(texto_empezar, boton_empezar)
        elif not jugada_realizada:
            # Mostrar la cuenta regresiva en la ventana de Pygame
            tiempo_restante = int(cuenta_regresiva - (time.time() - tiempo_inicio))
            if tiempo_restante > 0:
                fuente = pygame.font.Font(None, 100)
                texto = fuente.render(str(tiempo_restante), True, COLOR_TEXTO)
                pantalla.blit(texto, (ancho // 2 - texto.get_width() // 2, alto // 2 - texto.get_height() // 2))
            else:
                # Capturar el frame y realizar la predicción
                frame_procesado = capturar_puntos(frame)
                jugada = determinarJugada(frame_procesado, modelo)

                # Generar la jugada del bot
                jugada_bot = random.randint(0, 2)  # 0 = papel, 1 = piedra, 2 = tijera

                # Determinar el ganador
                resultado = determinarGanador(jugada, jugada_bot)
                if resultado == "Jugador gana":
                    puntaje_jugador += 1
                elif resultado == "IA gana":
                    puntaje_bot += 1

                jugada_realizada = True
                tiempo_inicio = time.time() + 2  # Esperar 2 segundos antes de reiniciar

                if primera_ronda:
                    cuenta_regresiva = 3
                    primera_ronda = False

        if jugada_realizada:
            # Mostrar la jugada del jugador
            fuente = pygame.font.Font(None, 50)
            texto_jugador = fuente.render(f"Jugador: {jugada}", True, COLOR_TEXTO)
            pantalla.blit(texto_jugador, (ancho - CUADRO_TAMANO - 50, alto // 2 + CUADRO_TAMANO // 2 + 10))

            # Mostrar la jugada del bot
            texto_bot = fuente.render(f"Bot: {CATEGORIES[jugada_bot]}", True, COLOR_TEXTO)
            pantalla.blit(texto_bot, (50, alto // 2 + CUADRO_TAMANO // 2 + 10))

            # Determinar el ganador
            resultado = determinarGanador(jugada, jugada_bot)
            texto_resultado = fuente.render(f"Resultado: {resultado}", True, COLOR_TEXTO)
            pantalla.blit(texto_resultado, (ancho // 2 - texto_resultado.get_width() // 2, alto - 50))
            
            # Mostrar el puntaje
            fuente = pygame.font.Font(None, 50)
            texto_puntaje = fuente.render(f"Jugador: {puntaje_jugador} - Bot: {puntaje_bot}", True, COLOR_TEXTO)
            pantalla.blit(texto_puntaje, (ancho // 2 - texto_puntaje.get_width() // 2, 50))

            # Verificar si alguien ganó
            if puntaje_jugador == modo // 2 + 1:
                mostrar_pantalla_final(pantalla, "¡Ganaste!")
                jugada_realizada = False
                return "menu"  # El jugador ganó
            elif puntaje_bot == modo // 2 + 1:
                mostrar_pantalla_final(pantalla, "Perdiste")
                jugada_realizada = False
                return "menu"  # El bot ganó
            
            if time.time() > tiempo_inicio:
                jugada_realizada = False
                tiempo_inicio = time.time()  # Reiniciar la cuenta regresiva

        pygame.display.flip()