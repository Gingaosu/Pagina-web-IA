from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('modelo.h5')

class_labels = ["Normal", "COVID-19"]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150), color_mode="grayscale") 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class
