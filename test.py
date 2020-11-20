import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        # Use a Rescaling layer to make sure input values are in the [0, 1] range.
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        # The original images have shape (28, 28), so we reshape them to (28, 28, 1)
        layers.Reshape(target_shape=(28, 28, 1)),
        # Follow-up with a classic small convnet
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=keras.metrics.SparseCategoricalAccuracy(),
)

model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)
