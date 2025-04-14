import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 50
IMG_WIDTH = 30
IMG_HEIGHT = 20
NUM_CATEGORIES = 3
TEST_SIZE = 0.3

CATEGORIES = [
    "papel",
    "piedra",
    "tijera",
]

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Train model
    model.fit(x_train,y_train, epochs=EPOCHS)

    # Evaluate
    model.evaluate(x_test, y_test, verbose=2)


def load_data(data_dir):
    """
    Load image data from directory `data_dir`

    Assume `data_dir` has one directory named after each category ()
    """
    images = []
    labels = []

    for i in range(NUM_CATEGORIES):
        print("Reading data: " + CATEGORIES[i])
        category_dir = os.path.join(data_dir, str(CATEGORIES[i]))
        for filename in os.listdir(category_dir):
            img_path = os.path.join(category_dir,filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0
            images.append(img)
            labels.append(i)

    return np.array(images), np.array(labels)

def get_model():
    model = tf.keras.Sequential([
        # Primera capa convolucional
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(20, 30, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Segunda capa convolucional
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Tercera capa convolucional
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),

        # Aplanar las características extraídas
        tf.keras.layers.Flatten(),

        # Capas densas
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),  # Dropout para evitar sobreajuste
        tf.keras.layers.Dense(3, activation="softmax")  # 3 clases de salida
    ])

    # Compile
    model.compile(
        optimizer="sgd",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()