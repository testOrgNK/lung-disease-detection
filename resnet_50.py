import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

# Load the ResNet-50 model pre-trained on ImageNet data
base_model = ResNet50(weights="imagenet", include_top=False)

# Add a Global Average Pooling 2D layer and a Dense layer for fine-tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(
    x
)  # Assuming 10 classes for demonstration

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Sample data for testing
x_train = np.random.rand(100, 224, 224, 3)  # Replace with your actual training data
y_train = np.random.randint(0, 10, size=(100, 10))  # Replace with your actual labels

# Train the model (for demonstration purposes)
model.fit(x_train, y_train, epochs=5, batch_size=32)
