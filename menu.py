import pygame
import sys
import random

def mostrar_menu():
    """
    Muestra el menú inicial con botones, un título y una lluvia de imágenes de piedra, papel o tijera.
    """
    # Inicializar pygame
    pygame.init()

    # Configurar la pantalla
    ancho, alto = 800, 600
    pantalla = pygame.display.set_mode((ancho, alto))
    pygame.display.set_caption("Menu - Piedra, Papel o Tijera")

    # Colores
    COLOR_FONDO = (153, 204, 255)  # Celeste
    COLOR_BOTON = (255, 255, 255)  # Blanco
    COLOR_TEXTO = (0, 0, 0)        # Negro

    # Fuentes
    fuente = pygame.font.Font(None, 50)
    fuente_titulo = pygame.font.Font(None, 80)  # Fuente más grande para el título

    # Botones
    botones = {
        "jugar": pygame.Rect(200, 150, 400, 50),
        "info": pygame.Rect(200, 250, 400, 50),
        "salir": pygame.Rect(200, 350, 400, 50),
    }

    # Cargar la imagen grande y dividirla en tres partes
    imagen_grande = pygame.image.load("piedra_papel_tijera.png")  # Asegúrate de usar el nombre correcto
    ancho_imagen = 1856  # Ancho de cada imagen individual
    alto_imagen = 1801   # Alto de cada imagen individual

    piedra = imagen_grande.subsurface((0, 0, ancho_imagen, alto_imagen))
    papel = imagen_grande.subsurface((ancho_imagen, 0, ancho_imagen, alto_imagen))
    tijera = imagen_grande.subsurface((ancho_imagen * 2, 0, ancho_imagen, alto_imagen))

    # Escalar las imágenes a un tamaño más pequeño
    piedra = pygame.transform.scale(piedra, (100, 100))
    papel = pygame.transform.scale(papel, (100, 100))
    tijera = pygame.transform.scale(tijera, (100, 100))

    # Lista de imágenes para la lluvia
    imagenes = [piedra, papel, tijera]

    # Lista para rastrear las posiciones de las imágenes
    lluvia = []

    # Estado del menú
    en_menu = True

    # Bucle principal
    while True:
        # Manejar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and en_menu:
                if botones["jugar"].collidepoint(event.pos):
                    en_menu = False  # Cambiar al estado del juego
                    return "jugar"
                elif botones["info"].collidepoint(event.pos):
                    print("Información del juego: Este es un juego de Piedra, Papel o Tijera.")
                elif botones["salir"].collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()

        # Dibujar fondo
        pantalla.fill(COLOR_FONDO)

        # Dibujar título del menú
        titulo = fuente_titulo.render("Piedra, Papel o Tijera", True, COLOR_TEXTO)
        pantalla.blit(titulo, (ancho // 2 - titulo.get_width() // 2, 50))  # Centrar el título horizontalmente

        # Generar nuevas imágenes para la lluvia
        if random.randint(0, 200) < 1:  # Reducir la frecuencia de aparición
            x = random.randint(0, ancho - 50)  # Posición horizontal aleatoria
            y = -50  # Comienza fuera de la pantalla (arriba)

            # Verificar que no haya otra imagen cerca en el eje x
            distancia_minima = 100
            posicion_valida = all(abs(x - existente[1]) > distancia_minima for existente in lluvia)

            if posicion_valida:
                img = random.choice(imagenes)  # Seleccionar una imagen aleatoria
                lluvia.append((img, x, y))

        # Dibujar y mover las imágenes de la lluvia
        for i, (img, x, y) in enumerate(lluvia):
            pantalla.blit(img, (x, y))
            lluvia[i] = (img, x, y + 0.125)  # Mover más lentamente hacia abajo

        # Eliminar imágenes que salieron de la pantalla
        lluvia = [(img, x, y) for img, x, y in lluvia if y < alto]

        if en_menu:
            # Dibujar botones
            for texto, rect in botones.items():
                pygame.draw.rect(pantalla, COLOR_BOTON, rect)
                texto_render = fuente.render(texto.capitalize(), True, COLOR_TEXTO)
                pantalla.blit(texto_render, (rect.x + 150, rect.y + 10))        

        # Actualizar pantalla
        pygame.display.flip()