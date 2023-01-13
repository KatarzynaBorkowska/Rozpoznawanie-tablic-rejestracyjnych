import os
import shutil

from flask import Flask, request, render_template
from glob import glob
from keras.utils import img_to_array
import cv2
import tensorflow as tf
import numpy as np
import pytesseract as pt
from matplotlib.pyplot import imsave

app = Flask(__name__)

model = tf.keras.models.load_model('./object_detection.h5')



def object_detection(path):

    image = tf.keras.utils.load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = tf.keras.utils.load_img(path, target_size=(224, 224))


    image_arr_224 = img_to_array(image1) / 255.0
    h, w, d = image.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)


    coords = model.predict(test_arr)


    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)


    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)


    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    return image, coords




@app.route('/', methods=['GET'])
def back_to_index():
    return render_template('index.html')


@app.route('/zdjecie_z_komputera', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save('../pythonProject/static/test.png')
    path = '../pythonProject/static/test.png'
    path2 = glob(path)
    print(path2)
    for path in path2:
        image, cods = object_detection(path)
        img = np.array(tf.keras.utils.load_img(path))
        xmin, xmax, ymin, ymax = cods[0]
        roi = img[ymin:ymax, xmin:xmax]

        text = pt.image_to_string(roi)
        imsave('static/image.jpg', roi)

    # Wyrenderuj szablon z przyciskiem powrotu
    return render_template('return.html', text=text, file_name=file.filename)

@app.route('/zdjecie_z_systemu', methods=['POST'])
def index_post():
    # Pobierz dane przesłane z formularza
    form_data = request.form['samochod']
    path = f'../pythonProject/static/{form_data}.png'
    path2 = glob(path)
    print(path2)
    for path in path2:
        image, cods = object_detection(path)
        img = np.array(tf.keras.utils.load_img(path))
        xmin, xmax, ymin, ymax = cods[0]
        roi = img[ymin:ymax, xmin:xmax]

        text = pt.image_to_string(roi)
        imsave('static/image.jpg', roi)
    # Skopiuj plik przed zmianą jego nazwy
    shutil.copy(path, '../pythonProject/static/test.png')
    # Wyrenderuj szablon z przyciskiem powrotu
    return render_template('return.html', text = text,image_name='test.png')



if __name__ == "__main__":
    app.run(debug=True)