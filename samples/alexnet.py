# Tensorflow v2.3.1

"""
Programmed by the-robot <https://github.com/the-robot>
"""

from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
)
from tensorflow.keras import Model
import tensorflow as tf
import typing

tf.config.run_functions_eagerly(True)

@tf.function
def AlexNet(input_shape: typing.Tuple[int], classes: int = 1000) -> Model:
    """
    Implementation of the AlexNet architecture.

    Arguments:
    input_shape -- shape of the images of the dataset
    classes     -- integer, number of classes

    Returns:
    model       -- a Model() instance in Keras

    Note:
    when you read the paper, you will notice that the channels (filters) in the diagram is only
    half of what I have written below. That is because in the diagram, they only showed model for
    one GPU (I guess for simplicity). However, during the ILSVRC, they run the network across 2 NVIDIA GTA 580 3GB GPUs.

    Also, in paper, they used Local Response Normalization. This can also be done in Keras with Lambda layer.
    You can also use BatchNormalization layer instead.
    """

    # convert input shape into tensor
    X_input = Input(input_shape)

    # NOTE: layer 1-5 is conv-layers
    # layer 1
    X = Conv2D(
        filters = 96,
        kernel_size = (11, 11),
        strides = (4, 4),
        activation = "relu",
        padding = "same",
    )(X_input)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)
    X = Lambda(tf.nn.local_response_normalization)(X)

    # layer 2
    X = Conv2D(
        filters = 256,
        kernel_size = (5, 5),
        strides = (1, 1),
        activation = "relu",
        padding = "same",
    )(X)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)
    X = Lambda(tf.nn.local_response_normalization)(X)

    # layer 3
    X = Conv2D(
        filters = 384,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = "relu",
        padding = "same",
    )(X)

    # layer 4
    X = Conv2D(
        filters = 384,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = "relu",
        padding = "same",
    )(X)

    # layer 5
    X = Conv2D(
        filters = 256,
        kernel_size = (3, 3),
        strides = (1, 1),
        activation = "relu",
        padding = "same",
    )(X)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)
    X = Lambda(tf.nn.local_response_normalization)(X)

    # NOTE: layer 6-7 is fully-connected layers
    # layer 6
    X = Flatten()(X)
    X = Dense(units = 2048, activation = 'relu')(X)
    X = Dropout(0.5)(X)

    # layer 7
    X = Dense(units = 2048, activation = 'relu')(X)
    X = Dropout(0.5)(X)

    # layer 8 (classification layer)
    # use sigmoid if binary classificaton and softmax if multiclass classification
    X = Dense(units = classes, activation = "softmax")(X)

    model = Model(inputs = X_input, outputs = X, name = "AlexNet")
    return model





from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import math


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

def return_loaded_batch(X, y, index, batch_size):
    out_X = []
    out_y = []
    if index >= len(X):
        batch_size = 0
        index = len(X) -1

    elif index + batch_size >= len(X):
        batch_size = len(X) - index - 1

    for item, cat in zip(X[index:index+batch_size], y[index:index+batch_size]):
        #print(cat)
        #print(item)
        out_X.append(cv2.imread(item))
        out_y.append(cat)
    
    yield (np.array(out_X), np.array(out_y))

#y_fishes = to_categorical(y_fishes)
X=np.array(X)
(x_train, y_train), (x_test, y_test) = (X[:70], y_fishes[:70]), (X[70:], y_fishes[70:])

# training parameters
batch_size = 2# orig paper trained all networks with batch_size=128
epochs = 500
data_augmentation = True
num_classes = 10
num_classes = len(np.unique(y_fishes))

# subtracting pixel mean improves accuracy
subtract_pixel_mean = True




# load the CIFAR10 data.
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimensions.
input_shape = x_train.shape[1:]

# normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# if subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

model = AlexNet(input_shape, num_classes)


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['acc'])
model.summary()

# enable this if pydot can be installed
# pip install pydot
#plot_model(model, to_file="%s.png" % model_type, show_shapes=True)

# prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % 'alexnet'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]


def fit_on_batch(model, X, y, batch_size, epochs=1):
    for epoch in range(epochs):
        index=0
        while index < len(X[:70]):
            for batch_x, batch_y in return_loaded_batch(X[:70], y, index, batch_size): #(x, y, batch_size):
                if batch_x.size == 0:
                    break
                model.train_on_batch(batch_x, batch_y)
                index+=batch_size 

        # score trained model
        scores = model.evaluate(x_test,
                                y_test,
                                batch_size=batch_size,
                                verbose=0)
        print('epoch', epoch)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

# run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
    
    #fit_on_batch(model, X_fishes, y_train, batch_size, epochs)
else:
    print('Using real-time data augmentation.')
    # this will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    steps_per_epoch =  math.ceil(len(x_train) / batch_size)
    # fit the model on the batches generated by datagen.flow().
    model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),
              verbose=1,
              epochs=epochs,
              validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)


# score trained model
scores = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])