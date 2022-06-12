from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import random

json_file = open('model.json','r')
model_json = json_file.read()
json_file.close()

model = keras.models.model_from_json(model_json)
model.load_weights("model.h5")

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from PIL import Image, ImageEnhance

IMAGE_SIZE = 30
BRIGHTNESS = (0.6, 1.4)
CONTRAST   = (0.6, 1.4)

def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(BRIGHTNESS[0],BRIGHTNESS[1]))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(CONTRAST[0],CONTRAST[1]))
    return image

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

class Model():
    def predict(c,file_path):
        test_images=open_images(file_path)
        y_pred=model.predict(test_images)
        y_pred=np.argmax(y_pred, axis=-1)
        return y_pred
        

