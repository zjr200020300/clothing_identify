from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import time
from flask import Flask
from cassandra.cluster import Cluster
import logging
log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)
cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
session = cluster.connect()
KEYSPACE = "mykeyspace"
app = Flask(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def createKeySpace():
    log.info("Creating keyspace...")
    try:
        session.execute("""
           CREATE KEYSPACE IF NOT EXISTS %s
           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
           """ % KEYSPACE)
        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)
        log.info("creating table...")
        session.execute("""
           CREATE TABLE mytable (
               time text,
               name text,
               perdict text,
               PRIMARY KEY (name)
           )
           """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)


createKeySpace()


def insertdata(fr, perdict):

    try:
        n = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        session.execute("""
            INSERT INTO mytable (time, name, perdict)
            VALUES ('%s', '%s', '%s')
            """ % (n, fr, perdict))
        log.info("%s, %s, %s" % (n, fr, perdict))
        log.info("Data stored!")
    except Exception as e:
        log.error("Unable to insert data!")
        log.error(e)


def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array, img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    a = class_names[predicted_label]
    return a


@app.route('/file/<file>', methods=['GET'])
def upload_file(file):
    global fr
    fr = '/Users/a912158001/Desktop/' + file

    def anal(fr):
        model = keras.models.load_model(
            '/Users/a912158001/PycharmProjects/image_classification/my_model.h5')
        test_images = Image.open(fr)
        test_images = np.invert(test_images.convert('L'))
        test_images = test_images / 255.0
        probability_model = tf.keras.Sequential([model,
                                                 tf.keras.layers.Softmax()])
        test_images = (np.expand_dims(test_images, 0))
        predictions_single = probability_model.predict(test_images)
        b = plot_image(0, predictions_single[0], test_images)
        return b
    perdict = anal(fr)
    insertdata(fr, perdict)
    return perdict
