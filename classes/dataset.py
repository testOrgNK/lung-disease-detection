import numpy as np
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from libs.location import *
from tensorflow.keras import Sequential, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class Dataset:
    def __init__(self) -> None:
        print("##Initialising Dataset")
        self.X = np.load(f"{DATASET_LOCATION_PATH}/X.npy")
        self.y = np.load(f"{DATASET_LOCATION_PATH}/y.npy")
        self.input_shape = (224, 224, 3)
        self.nClasses = len(np.unique(self.y))
        self.batch_size = 32
        self.epochs = 50
        self.model = None

    def create_model(self) -> Sequential:
        print("##Creating Custom Model")
        model = Sequential()
        model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=self.input_shape,
            )
        )
        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.nClasses, activation="softmax"))
        model.compile(
            optimizer="RMSprop", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model = model
        return self.model

    def create_chexnet_model(self, with_weights=True):
        print("##Creating Custom ChexNet Model")
        pre_model = DenseNet121(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )
        out = Dense(14, activation="sigmoid")(pre_model.output)
        chexnet_model = Model(inputs=pre_model.input, outputs=out)
        if with_weights:
            chexnet_model.load_weights(filepath=f"{CHEXNET_WEIGHTS_FILE}")
            for layer in chexnet_model.layers[:-2]:
                layer.trainable = False
        x = chexnet_model.layers[-2].output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.1)(x)
        output = Dense(2, activation="softmax")(x)
        chexnet_model_custom = Model(inputs=chexnet_model.input, outputs=output)
        chexnet_model.compile(
            optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model = chexnet_model_custom
        return self.model

    def compile(
        self, optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
    ):
        self.model.compile(optimizer, loss, metrics)

    def get_callbacks(self) -> None:
        stopping_callback = EarlyStopping(monitor="loss", patience=3)
        checkpoint_callback = ModelCheckpoint(
            filepath=f"{CHEXNET_MODEL_CHECKPOINT_LOCATION_PATH}/model_chexnet_{datetime.today().strftime('%Y-%m-%d')}.E{{epoch:02d}}-VL{{val_loss:.2f}}-A{{accuracy:.2f}}.h5",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
        return [stopping_callback, checkpoint_callback]

    def fit(self, model, X_train, X_test, y_train, y_test) -> None:
        print("##Fitting Model")
        start = time.time()
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_test, y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            callbacks=self.get_callbacks(),
        )
        end = time.time()
        print(f"Runtime of the program is {end - start}")
        return history
