import sys
from sklearn.model_selection import train_test_split

sys.path.insert(1, "/home/nitin/workspace/notebooks/pneumonia-detection")

from classes.dataset import Dataset
import pickle


DATASET_LOCATION_PATH = "/home/nitin/workspace/local_datasets/xray-dataset"
TRAINING_IMG_LOCATION_PATH = f"{DATASET_LOCATION_PATH}/stage_2_train_images"
CHECKPOINT_LOCATION_PATH = f"{DATASET_LOCATION_PATH}/checkpoints"
MODEL_CHECKPOINT_LOCATION_PATH = f"{CHECKPOINT_LOCATION_PATH}/model_checkpoints"

ds = Dataset()
model = ds.create_model()
X_train, X_test, y_train, y_test = train_test_split(
    ds.X, ds.y, test_size=0.2, random_state=42
)
model.compile(optimizer="RMSprop", loss="binary_crossentropy", metrics=["accuracy"])
history = ds.fit(model, X_train, X_test, y_train, y_test)
with open(f"{MODEL_CHECKPOINT_LOCATION_PATH}/trainHistoryDict", "wb") as file_pi:
    pickle.dump(history.history, file_pi)
