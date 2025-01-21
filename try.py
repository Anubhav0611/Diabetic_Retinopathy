from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import tensorflow as tf 
# Load an image for prediction

image_path = r"C:\Users\Anubhav\Desktop\Major_Project_2\venv\IDRiD_001.jpg"
img = image.load_img(image_path, target_size=IMAGE_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
model=tf.keras.models.load_model()
# Make predictions
predictions = model.predict(img_array)

# Interpret the predictions
class_index = np.argmax(predictions)
class_probability = predictions[0, class_index]

# Print the results
print(f"Predicted class index: {class_index}")
print(f"Predicted class probability: {class_probability}")