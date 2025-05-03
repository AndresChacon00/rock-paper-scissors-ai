import cv2
import numpy as np

def mostrar_menu():
    """
    Muestra el menú inicial con botones y devuelve la opción seleccionada.
    """
    def click_event(event, x, y, flags, param):
        """
        Maneja los clics del mouse en el menú.
        """
        nonlocal opcion
        if event == cv2.EVENT_LBUTTONDOWN:
            # Verificar si se hizo clic en el botón "Jugar"
            if 200 <= x <= 400 and 150 <= y <= 200:
                opcion = "jugar"
            # Verificar si se hizo clic en el botón "Info"
            elif 200 <= x <= 400 and 250 <= y <= 300:
                opcion = "info"
            # Verificar si se hizo clic en el botón "Salir"
            elif 200 <= x <= 400 and 350 <= y <= 400:
                opcion = "salir"

    opcion = None
    cv2.namedWindow("Menu")
    cv2.setMouseCallback("Menu", click_event)

    while True:
        # Crear una interfaz con fondo celeste
        interfaz = np.zeros((800, 600, 3), dtype=np.uint8)
        interfaz[:] = (255, 204, 153)  # Color celeste (BGR: 153, 204, 255)

        # Dibujar los botones
        cv2.rectangle(interfaz, (200, 150), (400, 200), (255, 255, 255), -1)  # Botón "Jugar"
        cv2.rectangle(interfaz, (200, 250), (400, 300), (255, 255, 255), -1)  # Botón "Info"
        cv2.rectangle(interfaz, (200, 350), (400, 400), (255, 255, 255), -1)  # Botón "Salir"

        # Agregar texto a los botones
        cv2.putText(interfaz, "Jugar", (240, 185), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(interfaz, "Info", (250, 285), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(interfaz, "Salir", (240, 385), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Mostrar la ventana
        cv2.imshow("Menu", interfaz)

        # Salir si se selecciona una opción
        if opcion is not None:
            cv2.destroyAllWindows()
            return opcion

        # Salir si se presiona 'Esc'
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Tecla 'Esc'
            cv2.destroyAllWindows()
            return "salir"