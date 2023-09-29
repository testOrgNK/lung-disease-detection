import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import time
from classes.dataset import Dataset
from libs.location import *

from sklearn.metrics import accuracy_score, classification_report

X_FILE_NAME = sys.argv[1]
Y_FILE_NAME = sys.argv[2]
WEIGHTS_FILE = sys.argv[3]

X = np.load(f"{DATASET_LOCATION_PATH}/{X_FILE_NAME}")
y = np.load(f"{DATASET_LOCATION_PATH}/{Y_FILE_NAME}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ds = Dataset()

chexnet_model = ds.create_chexnet_model(with_weights=False)
chexnet_model.load_weights(f"{CHEXNET_MODEL_CHECKPOINT_LOCATION_PATH}/{WEIGHTS_FILE}")
chexnet_model.compile(
    optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
)
y_pred = chexnet_model.predict(X_test)

y_train_final = [np.argmax(i) for i in y_train]
y_test_final = [np.argmax(i) for i in y_test]
y_pred_final = [np.argmax(i) for i in y_pred]

print(classification_report(y_pred_final, y_test_final))
