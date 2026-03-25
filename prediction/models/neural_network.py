import numpy as np
import tensorflow as tf
from tensorflow import keras
from features.config import RANDOM_SEED, N_CLASSES, TOTAL_FEATURE_COUNT


class NeuralNetworkModel:
    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self._model = None

    def _build(self) -> keras.Model:
        tf.random.set_seed(self.random_seed)
        inp = keras.Input(shape=(TOTAL_FEATURE_COUNT,))
        x = keras.layers.Dense(256, activation="relu")(inp)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3, seed=self.random_seed)(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.25, seed=self.random_seed)(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.2, seed=self.random_seed)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        out = keras.layers.Dense(N_CLASSES, activation="softmax")(x)
        model = keras.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y, X_val, y_val, epochs=200, batch_size=512, class_weight=None):
        self._model = self._build()
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, verbose=0),
        ]
        self._model.fit(X, y, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks, verbose=0,
                        class_weight=class_weight)
        return self

    def predict(self, X) -> np.ndarray:
        return np.argmax(self._model.predict(X, verbose=0), axis=1).astype(np.int32)

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict(X, verbose=0).astype(np.float32)

    def get_keras_model(self):
        return self._model

    def save(self, path: str) -> None:
        if not (path.endswith(".keras") or path.endswith(".h5")):
            path = path + ".keras"
        self._model.save(path)

    @classmethod
    def load(cls, path: str) -> "NeuralNetworkModel":
        if not (path.endswith(".keras") or path.endswith(".h5")):
            path = path + ".keras"
        instance = cls()
        instance._model = keras.models.load_model(path)
        return instance
