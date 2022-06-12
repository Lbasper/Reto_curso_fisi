# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 13:09:31 2022

@author: Lbasper
"""

import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageEnhance
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


# etiquetas = ['Karacadag', 'Basmati', 'Jasmine', 'Arborio', 'Ipsala']
etiquetas = ['Arborio','Basmati','Ipsala','Jasmine','Karacadag']
it_num=[0,1,2,3,4]

all_paths = []
Y_train = []


# a=5000
for label in range(len(etiquetas)):
    a=len(os.listdir("Rice_Image_Dataset/Train/"+etiquetas[label]))
    for i in range(1,a):
        all_paths.append("Rice_Image_Dataset/Train/{}/{} ({}).jpg".format(etiquetas[label],etiquetas[label],i))
        Y_train.append(etiquetas[label])
 
        
#b=500
#b=int((len(os.listdir(("Rice_Image_Dataset/Test")))-1)/5)

# new_paths= []
# Y_test =[]

# for label in range(len(etiquetas)):
#     for i in range(1,b):
#         new_paths.append("Rice_Image_Dataset/Test/{} ({}).jpg".format(etiquetas[label],10000+i))
#         Y_test.append(etiquetas[label])

x_train_paths=all_paths
# x_val_paths=new_paths
Y_train=Y_train
# Y_test=Y_test


BRIGHTNESS = (0.6, 1.4)
CONTRAST   = (0.6, 1.4)


def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(BRIGHTNESS[0],BRIGHTNESS[1]))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(CONTRAST[0],CONTRAST[1]))
    return image

def encode_labels(labels):
    encoded = []
    for x in labels:
        encoded.append(etiquetas.index(x))
    return np.array(encoded)

def decode_labels(labels):
    decoded = []
    for x in labels:
        decoded.append(etiquetas[x])
    return np.array(decoded)


IMAGE_SIZE = 30
def datagen(paths, labels, batch_size=12, epochs=3, augment=True):
    for _ in range(epochs):
        for x in range(0, len(paths), batch_size):
            batch_paths = paths[x:x+batch_size]
            batch_images = open_images(batch_paths, augment=augment)
            batch_labels = labels[x:x+batch_size]
            batch_labels = encode_labels(batch_labels)
            yield batch_images, batch_labels
            

def open_images(paths, augment=True):
    '''
    Given a list of paths to images, this function returns the images as arrays, and conditionally augments them
    '''
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE,IMAGE_SIZE))
        if augment:
            image = augment_image(image)
        image = np.array(image)/255.0
        images.append(image)
    return np.array(images)

train_images=open_images(x_train_paths)
train_labels=encode_labels(Y_train)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax),
    keras.layers.Dropout(0.01)
]
)


model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])


epochs = 10
steps = int(len(x_train_paths)/epochs)

model.fit(train_images,train_labels,epochs=epochs, steps_per_epoch=steps)

model_json = model.to_json()  
with open("model.json", "w") as json_file:  
    json_file.write(model_json)  
model.save("model.h5")