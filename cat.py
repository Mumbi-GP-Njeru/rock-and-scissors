# Create a model for the neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(
    train_image_generator,
    steps_per_epoch=total_train // BATCH_SIZE,
    epochs=50,
    validation_data=validation_image_generator,
    validation_steps=total_validation // BATCH_SIZE
)
# Make predictions on the test images
import numpy as np

test_images = test_image_generator.next()[0]
test_labels = test_image_generator.next()[1]

predictions = model.predict(test_images)
probabilities = np.round(predictions)

# Plot the images and their predictions
plotImages(test_images, probabilities)