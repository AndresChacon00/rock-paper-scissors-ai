from menu import mostrar_menu
from juego import iniciar_juego

opcion = mostrar_menu()

if opcion == "jugar":
    iniciar_juego()
elif opcion == "info":
    print("Información del juego: Este es un juego de Piedra, Papel o Tijera.")
elif opcion == "salir":
    print("Saliendo del juego...")
    exit()