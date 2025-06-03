# rock-paper-scissors-ai
Proyecto para la materia de inteligencia artificial

El modelo de inteligencia artificial reconoce a partir de la webcam el gesto del usuario (piedra, papel o tijeras).
Utiliza los puntos de la mano para determinar el gesto, en vez de un modelo convolucional.

Datos obtenidos de https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors?resource=download
Los cuales fueron preprocesados quitando el fondo a las imágenes y colocandolas en blanco y negro

https://www.kaggle.com/datasets/alexandredj/rock-paper-scissors-dataset


USO
La carpeta de los datos puede tener cualquier nombre pero las imágenes deben estar ordenadas en 3 carpetas llamadas exactamente
"piedra", "papel" y "tijera"
 `python model_generator.py {data_folder_name}`
