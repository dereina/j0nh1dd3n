'''Implements a Y-Network using Functional API
~99.3% test accuracy
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

#own imports
import cv2
import glob
from os import walk
from os.path import join

#utils


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p.astype(int)], b[p.astype(int)]



#configs
mypath = 'images' #will load folders as classes
f = []
label_encoder= {}
X_fishes = []
y_fishes = []
i=0
for (dirpath, dirnames, filenames) in walk(mypath):
    for dir_name in dirnames:
        label_encoder[dir_name]=i
        i+=1
        for file in glob.glob(join(dirpath, dir_name)+"/*.png"):
            print(file)
            #dict_images_X[dir_name].append(file)
            X_fishes.append(file)
            y_fishes.append(label_encoder[dir_name])
    f.extend(filenames)
    break

X_fishes, y_fishes = unison_shuffled_copies(np.array(X_fishes), np.array(y_fishes))
X=[]
for item, cat in zip(X_fishes, y_fishes):
     print(cat)
     print(item)
     X.append(cv2.imread(item))

#y_fishes = to_categorical(y_fishes)
X=np.array(X)
(x_train, y_train), (x_test, y_test) = (X[:70], y_fishes[:70]), (X[70:], y_fishes[70:])

# load MNIST dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# from sparse label to categorical
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# reshape and normalize input images
image_size = x_train.shape[1]
#x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
#x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = x_train.shape[1:]#(image_size, image_size, 1)
batch_size = 4
kernel_size = 11
dropout = 0.4
n_filters = 32

# left branch of Y network
left_inputs = Input(shape=input_shape)
x = left_inputs
filters = n_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for i in range(3):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2

# right branch of Y network
right_inputs = Input(shape=input_shape)
y = right_inputs
filters = n_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for i in range(3):
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu',
               dilation_rate=2)(y)
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2

# merge left and right branches outputs
y = concatenate([x, y])
# feature maps to vector before connecting to Dense 
y = Flatten()(y)
y = Dropout(dropout)(y)
outputs = Dense(num_labels, activation='softmax')(y)

# build the model in functional API
model = Model([left_inputs, right_inputs], outputs)

# verify the model using graph
# enable this if pydot can be installed
# pip install pydot
#plot_model(model, to_file='cnn-y-network.png', show_shapes=True)

# verify the model using layer text description
model.summary()

# classifier loss, Adam optimizer, classifier accuracy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train the model with input images and labels
model.fit([x_train, x_train],
          y_train, 
          validation_data=([x_test, x_test], y_test),
          epochs=20,
          batch_size=batch_size)

# model accuracy on test dataset
score = model.evaluate([x_test, x_test],
                       y_test,
                       batch_size=batch_size,
                       verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))