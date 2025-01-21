# app.py
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ReLU
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #interacting with input and output directories
import tensorflow as tf #framework for creating the neural network
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import PIL #Python Imaging Library for image manipulations
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt # matplot for plotting graphs and displaying sample images

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
path = r'C:\Users\gargg\Desktop\prdata\train'

csv_path = r'C:\Users\gargg\Desktop\prdata\ten.csv'
df = pd.read_csv(csv_path)

def rename_to_path(image_name):
    return f'C:/Users/gargg/Desktop/prdata/train/{image_name}.png'

# Apply the function to the 'image_names' column
df['Image name'] = df['Image name'].apply(rename_to_path)
df['Retinopathy grade'] = df['Retinopathy grade'].astype(str)

from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=24)

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Image name',
    y_col='Retinopathy grade',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='Image name',
    y_col='Retinopathy grade',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Plot accuracy graph
model=load_model('Inception.h5')
history = model.history.history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Generate predictions
y_pred = model.predict(valid_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert one-hot encoded labels to actual labels
y_true = np.argmax(valid_generator.labels, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=train_generator.class_indices.keys(), yticklabels=train_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print classification report
class_labels = list(train_generator.class_indices.keys())
print(classification_report(y_true, y_pred_classes, target_names=class_labels))
