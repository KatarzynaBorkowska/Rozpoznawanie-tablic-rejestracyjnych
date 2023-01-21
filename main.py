import os
from glob import glob
import xml.etree.ElementTree as xet
import pandas as pd
from keras.utils import img_to_array
import cv2
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


path = glob('../pythonProject/archive/annotations/*.xml')
labels_dict = dict(filepath=[], xmin=[], xmax=[], ymin=[], ymax=[])

# odczytywanie wartości z pliku xml
for filename in path:
    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

df = pd.DataFrame(labels_dict)
df.to_csv('wartosci.csv', index=False)
df.head()

filename = df['filepath'][0]


def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('../pythonProject/archive/images/', filename_image)
    return filepath_image


getFilename(filename)
image_path = list(df['filepath'].apply(getFilename))

# Celujemy wszystkie nasze wartości w tablicy wybierając wszystkie kolumny
labels = df.iloc[:, 1:].values
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h, w, d = img_arr.shape

    load_image = tf.keras.utils.load_img(image, target_size=(224, 224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr / 255.0  # znormalizowanie

    xmin, xmax, ymin, ymax = labels[ind]
    nxmin, nxmax = xmin / w, xmax / w
    nymin, nymax = ymin / h, ymax / h
    label_norm = (nxmin, nxmax, nymin, nymax)  # Wyjście znormalizowane

    data.append(norm_load_image_arr)
    output.append(label_norm)

# konwertuj dane na tablice
X = np.array(data, dtype=np.float32)
y = np.array(output, dtype=np.float32)

# ML

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

inception_resnet = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights="imagenet", include_top=False,
                                                                               input_tensor=tf.keras.layers.Input(
                                                                                   shape=(224, 224, 3)))
# ---------------------
headmodel = inception_resnet.output
headmodel = tf.keras.layers.Flatten()(headmodel)
headmodel = tf.keras.layers.Dense(500, activation="relu")(headmodel)
headmodel = tf.keras.layers.Dense(250, activation="relu")(headmodel)
headmodel = tf.keras.layers.Dense(4, activation='sigmoid')(headmodel)

# ---------- model

model = tf.keras.models.Model(inputs=inception_resnet.input, outputs=headmodel)

model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()

tfb = tf.keras.callbacks.TensorBoard('object_detection')
history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=150,
                    validation_data=(x_test,y_test),callbacks=[tfb])

model.save('./object_detection.h5')
