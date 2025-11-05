# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import matplotlib.pyplot as plt 

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")


(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()


# visualizaiton
class_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

fig, axes = plt.subplots(1, 5, figsize=(15, 10))

for i in range(5):
    axes[i].imshow(xtrain[i])
    label = class_labels[int(ytrain[i])]
    axes[i].set_title(label)
    axes[i].axis("off")

plt.show()

#data normalization 
xtrain = xtrain.astype("float32")/255
xtest = xtest.astype("float32")/255

#one - hot encoding
ytrain = to_categorical(ytrain,10)
ytest = to_categorical(ytest,10)


#Data Augmentation 
datagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest"
    )

datagen.fit(xtrain)























