import cv2
import mediapipe as mp

class detectorManos():
    def __init__(self, mode=False, maxManos=1, model_complexity=1, confDeteccion=0.5, confSeguimiento=0.5):
        self.mode = mode
        self.maxManos = maxManos
        self.compl = model_complexity
        self.confDeteccion = confDeteccion
        self.confSeguimiento = confSeguimiento

        self.mpManos = mp.solutions.hands
        self.manos = self.mpManos.Hands(self.mode, self.maxManos, self.compl, self.confDeteccion, self.confSeguimiento)
        self.dibujo = mp.solutions.drawing_utils

    def encontrarManos(self, frame, dibujar=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(frameRGB)

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame, mano, self.mpManos.HAND_CONNECTIONS)
        return frame

    def encontrarPosicion(self, frame, ManoNum = 0, dibujar = True, color = []):
        xlista = []
        ylista = []
        bbox = []
        player = 0
        self.lista = []
        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[ManoNum]
            prueba = self.resultados.multi_hand_landmarks
            player = len(prueba)
            #print(player)
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape  # Extraemos las dimensiones de los fps
                cx, cy = int(lm.x * ancho), int(lm.y * alto)  # Convertimos la informacion en pixeles
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id, cx, cy])
                if dibujar:
                    cv2.circle(frame,(cx, cy), 3, (0, 0, 0), cv2.FILLED)  # Dibujamos un circulo

            xmin, xmax = min(xlista), max(xlista)
            ymin, ymax = min(ylista), max(ylista)
            bbox = xmin, ymin, xmax, ymax
            if dibujar:
                # Dibujamos cuadro
                cv2.rectangle(frame,(xmin - 20, ymin - 20), (xmax + 20, ymax + 20), color,2)
        return self.lista, bbox, player


def main():
    cap = cv2.VideoCapture(0)
    detector = detectorManos()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.encontrarManos(frame, dibujar=False)
        bbox = detector.encontrarPosicion(frame)

        cv2.imshow("Manos", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()