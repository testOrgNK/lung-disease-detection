WORKSPACE_PATH = "/home/nitin/workspace"
DATASETS_LOCATION_PATH = f"{WORKSPACE_PATH}/local_datasets"
MODELS_CHECKPOINT_LOCATION_PATH = f"{WORKSPACE_PATH}/models"
DATASET_LOCATION_PATH = f"{DATASETS_LOCATION_PATH}/xray-dataset"
TRAINING_IMG_LOCATION_PATH = f"{DATASET_LOCATION_PATH}/stage_2_train_images"
TRAINING_IMG_JPG_LOCATION_PATH = (
    f"{WORKSPACE_PATH}/notebooks/pneumonia-detection/stage_2_train_images_jpg"
)
BALANCED_TRAINING_IMG_JPG_LOCATION_PATH = (
    f"{WORKSPACE_PATH}/notebooks/pneumonia-detection/balanced_stage_2_train_images_jpg"
)
CLASSIFIED_TRAINING_IMG_JPG_LOCATION_PATH = f"{WORKSPACE_PATH}/notebooks/pneumonia-detection/classified_stage_2_train_images_jpg"
CHECKPOINT_LOCATION_PATH = f"{DATASET_LOCATION_PATH}/checkpoints"
CHEXNET_MODEL_CHECKPOINT_LOCATION_PATH = f"{MODELS_CHECKPOINT_LOCATION_PATH}/xray"
CHEXNET_WEIGHTS_FILE = f"{CHEXNET_MODEL_CHECKPOINT_LOCATION_PATH}/brucechou1983_CheXNet_Keras_0.3.0_weights.h5"