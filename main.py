from menu import mostrar_menu
from juego import iniciar_juego
import pygame
import sys

def main():
    pygame.init()
    ancho, alto = 1000, 600  # Aumentar el tamaño de la pantalla
    pantalla = pygame.display.set_mode((ancho, alto))
    pygame.display.set_caption("Piedra, Papel o Tijera")

    estado = "menu"  # Estado inicial

    while True:
        if estado == "menu":
            estado = mostrar_menu(pantalla)
        elif estado == "jugar":
            estado = iniciar_juego(pantalla)
        elif estado == "salir":
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    main()