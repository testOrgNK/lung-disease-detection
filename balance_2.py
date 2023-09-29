from PIL import Image, ImageEnhance
import os
from libs.location import *

# Define the paths
input_folder = f"{CLASSIFIED_TRAINING_IMG_JPG_LOCATION_PATH}/class-1"
output_folder = BALANCED_TRAINING_IMG_JPG_LOCATION_PATH

# Create a list of original file paths
original_files = os.listdir(input_folder)

# Define data augmentation parameters
# You can adjust these parameters as needed
rotation_degrees = 15
flip_horizontal = True
brightness_factor = 1.2
target_size = (224, 224)  # Specify the target size for resizing


# Specify the number of augmented versions per original image
augmentation_factor = 3  # You can adjust this as needed

# Iterate through original images and generate augmented versions
for i, file_name in enumerate(original_files):
    original_path = os.path.join(input_folder, file_name)
    original_image = Image.open(original_path)

    for j in range(augmentation_factor):
        augmented_image = original_image.copy()
        augmented_image = augmented_image.resize(target_size)

        # Apply data augmentation operations here
        augmented_image = augmented_image.rotate(rotation_degrees)
        if flip_horizontal:
            augmented_image = augmented_image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_image = ImageEnhance.Brightness(augmented_image).enhance(
            brightness_factor
        )

        # Generate a new file name with an index
        new_file_name = (
            f"{os.path.splitext(file_name)[0]}_{i * augmentation_factor + j + 1}.jpg"
        )
        new_file_path = os.path.join(output_folder, new_file_name)

        # Save the augmented image
        augmented_image.save(new_file_path)

print("Augmentation complete.")
