import os
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from libs.location import *
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

dataset_dir = CLASSIFIED_TRAINING_IMG_JPG_LOCATION_PATH
output_dir = BALANCED_TRAINING_IMG_JPG_LOCATION_PATH

target_class_balance = 1.0
# for filename in os.listdir(dataset_dir):
#     class_name = filename.split("_")[
#         0
#     ]  # Extract class name from filename (assuming filenames are like "class_1.jpg")
#     class_dir = os.path.join(dataset_dir, class_name)
#     os.makedirs(class_dir, exist_ok=True)
#     shutil.move(os.path.join(dataset_dir, filename), os.path.join(class_dir, filename))


# # Step 2: Balance each class individually
# for class_name in os.listdir(dataset_dir):
#     class_dir = os.path.join(dataset_dir, class_name)
#     print(class_dir)
#     num_original_samples = len(os.listdir(class_dir))
#     num_samples_needed = int(target_class_balance * num_original_samples)
#     files_in_class = os.listdir(class_dir)
#     # print(files_in_class)
#     datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode="nearest",
#         brightness_range=[-1, 1],
#     )
#     output_class_dir = os.path.join(output_dir, class_name)

#     os.makedirs(output_class_dir, exist_ok=True)

#     try:
#         print(f"Augmenting images for class: {class_name}")
#         class_datagen = datagen.flow_from_directory(
#             dataset_dir,
#             target_size=(224, 224),
#             batch_size=32,
#             save_to_dir=output_class_dir,
#             save_format="jpg",
#             classes=[class_name],  # Specify the class name
#         )

#         for _ in range(num_samples_needed // 32):
#             batch = class_datagen.next()

#     except Exception as e:
#         print(f"Error: {str(e)}")


# # # Optional Step 3: Reorganize the dataset back into a single folder if needed
# # if output_dir != dataset_dir:
# #     for class_name in os.listdir(output_dir):
# #         class_dir = os.path.join(output_dir, class_name)
# #         for filename in os.listdir(class_dir):
# #             shutil.move(
# #                 os.path.join(class_dir, filename), os.path.join(output_dir, filename)
# #             )
# #         os.rmdir(class_dir)

# # Your balanced dataset is now created in the 'output_dir'


# Calculate the target number of samples for each class
target_num_samples = {}

# Loop through the class directories to count the number of samples in each class
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    num_samples = len(os.listdir(class_dir))
    target_num_samples[class_name] = num_samples

# Find the class with fewer samples (minority class)
minority_class = min(target_num_samples, key=target_num_samples.get)
majority_class = max(target_num_samples, key=target_num_samples.get)
minority_samples = target_num_samples[minority_class]
majority_samples = target_num_samples[majority_class]

difference_in_samples = (
    target_num_samples[majority_class] - target_num_samples[minority_class]
)

# Initialize the ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    brightness_range=[-1, 1],
)

# Loop through the class directories to balance the dataset
class_name = minority_class
class_dir = os.path.join(dataset_dir, class_name)

# Calculate the number of additional samples needed for this class
num_samples_needed = majority_samples

print(num_samples_needed)
# Create an output directory for the current class
output_class_dir = os.path.join(output_dir, class_name)
os.makedirs(output_class_dir, exist_ok=True)

try:
    print(f"Augmenting images for class: {class_name}")
    class_datagen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        classes=[class_name],
        save_to_dir=output_class_dir,
        save_format="jpg",
        shuffle=False,
    )

    counter = 0
    # Generate augmented samples until the target balance is reached
    for i in range(num_samples_needed // 32):
        batch = class_datagen.next()
        for j, image_array in enumerate(batch):
            original_name = class_datagen.filenames[j].split(os.path.sep)[-1]
            print(original_name)
            new_name = f"{original_name[:-4]}_{counter}.jpg"
            counter += 1
            print(new_name)
            image_path = os.path.join(output_class_dir, original_name)
            print(image_path)
            new_image_path = os.path.join(output_class_dir, new_name)
            print(new_image_path)
            os.rename(image_path, new_image_path)

except Exception as e:
    print(f"Error: {str(e)}")
