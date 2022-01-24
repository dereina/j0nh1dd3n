# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# convert sparse label to categorical values
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# preprocess the input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# parameters for the network
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3



# utiliaing functional API to build cnn layers
inputs = Input(shape=input_shape)
y = Conv2D(filters=filters,
    kernel_size=kernel_size,
    activation='relu')(inputs)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
    kernel_size=kernel_size,
    activation='relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
    kernel_size=kernel_size,
    activation='relu')(y)
# convert image to vector 
y = Flatten()(y)
# dropout regularization
y = Dropout(dropout)(y)
y = Dense(256, activation='softmax')(y)
outputs = Dense(num_labels, activation='softmax')(y)
    # model building by supplying inputs/outputs
model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy',
	 optimizer='adam',
	 metrics=['accuracy'])
model.fit(x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=1,
    batch_size=batch_size)
# accuracy evaluation
score = model.evaluate(x_test,
    y_test,
    batch_size=batch_size,
    verbose=0)
print("nTest accuracy: %.1f%%" % (100.0 * score[1]))