#importing libraries 
from __future__ import division, print_function
import sys
import os
import glob
import re
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from keras.preprocessing.image import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_ngrok import run_with_ngrok

# Define a flask app
app = Flask(__name__)
run_with_ngrok(app)
#providing path for the model
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()

#Prediction function
def model_predict(img_path, model):
    
    img = image.load_img(img_path, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x) 
    return preds
  

@app.route('/', methods=['GET'])
def index():
    # Main page which will be visible 
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        result = str(preds)# Convert to string
        #return result
        if result=='[[0.]]':
            return 'Bird'
        else:
            return 'Drone'
    return None

# running the flask app
if __name__ == '__main__':
    app.run()

