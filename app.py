# app.py
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ReLU
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

class CustomReLU(ReLU):
    def __init__(self, **kwargs):
        super(CustomReLU, self).__init__(**kwargs)

app = Flask(__name__)
model = load_model('Inception.h5')
IMAGE_SIZE = (224, 224)
def process_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size = IMAGE_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img_test = img/255.0
    pred = model.predict(img_test)
    pred = np.argmax(pred)
    return pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    img_path = 'temp_image.jpg'
    file.save(img_path)

    prediction = process_image(img_path)
    # prediction = model.predict(img_path)
    print("prediction")
    print(prediction)
    
    
    
    # max_index = np.array(result).argmax()
    
    
    # result_list = result.tolist()

    return jsonify({'result': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
