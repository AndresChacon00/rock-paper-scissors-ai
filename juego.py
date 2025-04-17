import cv2
import seguimiento_manos as sm
import tensorflow as tf
import numpy as np
import time

IMG_WIDTH = 90
IMG_HEIGHT = 60

CATEGORIES = [
    "papel",
    "piedra",
    "tijera",
]

cap = cv2.VideoCapture(0)
detector = sm.detectorManos(confDeteccion=0.9)

# Variables para rastrear el movimiento
subidas = 0  # Contador de subidas de la mano
mano_arriba = False  # Estado de la mano (si está arriba o no)
jugada_realizada = False # Controla si ya se realizó la prediccion
tiempo_espera = 0


n_juegos=3
juego_actual=1

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
    
def determinarJugada(frame, model):
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = frame / 255.0

    frame = np.expand_dims(frame, axis=0)
    prediccion = model.predict(frame, verbose=0)

    jugada = np.argmax(prediccion)
    return jugada

# Cargar modelo
modelo = tf.keras.models.load_model("rock_paper_scissors_model.h5")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    alto, ancho, c = frame.shape
    mitad_frame = alto // 2  # Calcular la mitad del frame
    frame = cv2.flip(frame, 1)

    # Dibujar una línea en la mitad del frame
    cv2.line(frame, (0, mitad_frame), (ancho, mitad_frame), (0, 255, 0), 2)

    # Encontrar manos
    frame = detector.encontrarManos(frame, dibujar=True)
    posiciones, _, _ = detector.encontrarPosicion(frame, dibujar=False)

    if posiciones:
        # Obtener la coordenada y del punto clave de la muñeca (id=0)
        for punto in posiciones:
            if punto[0] == 0:  # id=0 corresponde a la muñeca
                _, _, y = punto
                break

        # Detectar si la mano sube más allá de la mitad del frame
        if y < mitad_frame:  
            if not mano_arriba:  
                subidas += 1
                mano_arriba = True  
                
        else:
            mano_arriba = False  

        # Mostrar el número de subidas detectadas
        if subidas <= 3:
            cv2.putText(frame, f"{subidas}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if subidas == 3 and mano_arriba==False and not jugada_realizada:
            cv2.putText(frame, "YA", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            tiempo_espera = time.time() + 2
            jugada = 0#determinarJugada(frame, modelo)
    
            jugada_realizada = True

    if jugada_realizada:
        cv2.putText(frame, f"Jugada: {CATEGORIES[jugada]}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if jugada_realizada and time.time() > tiempo_espera:
        subidas = 0
        jugada_realizada = False

    # Mostrar el frame
    cv2.imshow("Piedra, Papel o Tijera", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()