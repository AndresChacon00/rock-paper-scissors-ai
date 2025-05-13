import pygame
import sys
import random

# Colores
COLOR_FONDO = (153, 204, 255)  # Celeste
COLOR_BOTON = (255, 255, 255)  # Blanco
COLOR_TEXTO = (0, 0, 0)        # Negro

def mostrar_menu(pantalla):
    """
    Muestra el menú inicial con botones, un título y una lluvia de imágenes de piedra, papel o tijera.
    """
    ancho, alto = pantalla.get_size()    

    # Fuentes
    fuente = pygame.font.Font(None, 50)
    fuente_titulo = pygame.font.Font(None, 80)

    # Dimensiones de los botones
    boton_ancho = 400
    boton_alto = 50
    espacio_entre_botones = 30

    # Posiciones centradas para los botones
    botones = {
        "jugar": pygame.Rect((ancho - boton_ancho) // 2, alto // 2 - boton_alto - espacio_entre_botones, boton_ancho, boton_alto),
        "info": pygame.Rect((ancho - boton_ancho) // 2, alto // 2, boton_ancho, boton_alto),
        "salir": pygame.Rect((ancho - boton_ancho) // 2, alto // 2 + boton_alto + espacio_entre_botones, boton_ancho, boton_alto),
    }

    # Cargar la imagen grande y dividirla en tres partes
    imagen_grande = pygame.image.load("piedra_papel_tijera.png")  # Asegúrate de usar el nombre correcto
    ancho_imagen = 1856  # Ancho de cada imagen individual
    alto_imagen = 1801   # Alto de cada imagen individual

    tijera = imagen_grande.subsurface((0, 0, ancho_imagen, alto_imagen))
    piedra = imagen_grande.subsurface((ancho_imagen, 0, ancho_imagen, alto_imagen))
    papel = imagen_grande.subsurface((ancho_imagen * 2, 0, ancho_imagen, alto_imagen))

    # Escalar las imágenes a un tamaño más pequeño
    piedra = pygame.transform.scale(piedra, (100, 100))
    papel = pygame.transform.scale(papel, (100, 100))
    tijera = pygame.transform.scale(tijera, (100, 100))

    # Lista de imágenes para la lluvia
    imagenes = [piedra, papel, tijera]

    # Lista de imágenes para la lluvia
    lluvia = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "salir"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if botones["jugar"].collidepoint(event.pos):
                    return "jugar"
                elif botones["info"].collidepoint(event.pos):
                    estado = mostrar_informacion(pantalla)
                    if estado == "salir":
                        return "salir"
                elif botones["salir"].collidepoint(event.pos):
                    return "salir"

        pantalla.fill(COLOR_FONDO)

        # Llamar a la función de lluvia
        lluvia_imagenes(pantalla, lluvia, imagenes, ancho, alto)

        # Dibujar título
        titulo = fuente_titulo.render("Piedra, Papel o Tijera", True, COLOR_TEXTO)
        pantalla.blit(titulo, (ancho // 2 - titulo.get_width() // 2, alto // 4 - titulo.get_height() // 2))        

        # Dibujar botones con bordes
        for texto, rect in botones.items():
            borde_rect = rect.inflate(4, 4)  # Crear un borde ligeramente más grande
            pygame.draw.rect(pantalla, (0, 0, 0), borde_rect)  # Dibujar el borde negro
            pygame.draw.rect(pantalla, COLOR_BOTON, rect)  # Dibujar el botón blanco
            texto_render = fuente.render(texto.capitalize(), True, COLOR_TEXTO)
            pantalla.blit(texto_render, (rect.x + (rect.width - texto_render.get_width()) // 2, rect.y + 10))

        pygame.display.flip()

def lluvia_imagenes(pantalla, lluvia, imagenes, ancho, alto):
    """
    Dibuja y actualiza la lluvia de imágenes en la pantalla.

    Parámetros:
    - pantalla: Superficie de Pygame donde se dibujará la lluvia.
    - lluvia: Lista de imágenes en movimiento.
    - imagenes: Lista de imágenes disponibles para la lluvia.
    - ancho: Ancho de la pantalla.
    - alto: Alto de la pantalla.
    """
    # Generar nuevas imágenes para la lluvia
    if random.randint(0, 200) < 1:  # Reducir la frecuencia de aparición
        x = random.randint(0, ancho - 50)  # Posición horizontal aleatoria
        y = -50  # Comienza fuera de la pantalla (arriba)

        # Verificar que no haya otra imagen cerca en el eje x
        distancia_minima = 100
        posicion_valida = all(abs(x - existente[1]) > distancia_minima for existente in lluvia)

        if posicion_valida:
            img = random.choice(imagenes)  # Seleccionar una imagen aleatoria
            img_escalada = pygame.transform.scale(img, (100, 100))  # Escalar la imagen a 100x100
            lluvia.append((img_escalada, x, y))

    # Dibujar y mover las imágenes de la lluvia
    for i, (img, x, y) in enumerate(lluvia):
        pantalla.blit(img, (x, y))
        lluvia[i] = (img, x, y + 0.125)  # Velocidad de caída

    # Eliminar imágenes que salieron de la pantalla
    lluvia[:] = [(img, x, y) for img, x, y in lluvia if y < alto]

def mostrar_informacion(pantalla):
    """
    Muestra una pantalla con información sobre el juego.
    """
    ancho, alto = pantalla.get_size()

    # Fuentes
    fuente = pygame.font.Font(None, 50)
    fuente_titulo = pygame.font.Font(None, 80)

    # Título
    titulo = fuente_titulo.render("Información del Juego", True, COLOR_TEXTO)

    # Texto de información
    texto_lineas = [
        "Este es un juego de Piedra, Papel o Tijera.",
        "Reglas:",
        "- Piedra vence a Tijera.",
        "- Tijera vence a Papel.",
        "- Papel vence a Piedra.",
        "",
        "El juego detecta tu mano por medio de la cámara.",
        "Tiene dos modos de juego:",
        "- 2 de 3: Gana el primero en ganar 2 rondas.",
        "- 3 de 5: Gana el primero en ganar 3 rondas.",
        "",
        "¡Diviértete jugando contra la IA!",
    ]

    # Botón para regresar al menú
    boton_volver = pygame.Rect((ancho - 200) // 2, alto - 100, 200, 50)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "salir"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if boton_volver.collidepoint(event.pos):
                    return "menu"

        pantalla.fill(COLOR_FONDO)

        # Dibujar el título
        pantalla.blit(titulo, (ancho // 2 - titulo.get_width() // 2, 50))

        # Dibujar el texto de información
        for i, linea in enumerate(texto_lineas):
            texto = fuente.render(linea, True, COLOR_TEXTO)
            pantalla.blit(texto, (ancho // 2 - texto.get_width() // 2, 150 + i * 40))

        # Dibujar el botón "Volver"
        pygame.draw.rect(pantalla, (0, 0, 0), boton_volver.inflate(4, 4))  # Borde negro
        pygame.draw.rect(pantalla, COLOR_BOTON, boton_volver)  # Botón blanco
        texto_volver = fuente.render("Volver", True, COLOR_TEXTO)
        pantalla.blit(texto_volver, (boton_volver.x + (boton_volver.width - texto_volver.get_width()) // 2,
                                     boton_volver.y + (boton_volver.height - texto_volver.get_height()) // 2))

        pygame.display.flip()