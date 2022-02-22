from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from json import load
from numpy.core.numeric import True_

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

import tensorflow.keras as keras
from tensorflow.keras.models import load_model

#own imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import cv2
import glob
from os import walk
from os.path import join
import numpy as np

from sklearn.metrics import confusion_matrix
import tensorflow as tf

import argparse


#utils
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


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p.astype(int)], b[p.astype(int)]


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

def fit_on_batch(model, X, y, x_test, y_test, batch_size, split, epochs=1):
    for epoch in range(epochs):
        index=0
        while index < len(X[:split]):
            for batch_x, batch_y in return_loaded_batch(X[:split], y, index, batch_size): #(x, y, batch_size):
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


def confusion_matrix(model, y_test, x_test, output):
    #confusion matrix
    labels = np.argmax(y_test, axis=1)
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)
    #confusion_matrix = confusion_matrix(labels, predictions)
    #confusion_matrix(predicted_categories, true_categories)
    confusion_mtx = tf.math.confusion_matrix(labels=labels, predictions=predictions)
    print(confusion_mtx)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                #xticklabels=commands,
                #yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.savefig(output)
    plt.close()
    #plt.show()

class TransferLearning():
    def __init__(self, images_path, train_test_split=0.80, batch_size=2, batch_size_ft=2, epochs=200, epochs_ft=10, max_items_per_class = 1000, hidden_layer_size=128, data_augmentation=True, substract_pixel_mean=False, model_path='', save_model_name='model.h5', save_weights_name='weights.h5' ):
        self.images_path = images_path
        self.model_path = model_path
        self.save_model_name = save_model_name
        self.save_weights_name = save_weights_name

        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.batch_size_ft = batch_size_ft
        self.epochs = epochs
        self.epochs_ft = epochs_ft
        self.max_items_per_class = max_items_per_class
        self.hidden_layer_size = hidden_layer_size
        self.data_augmentation = data_augmentation
        # subtracting pixel mean improves accuracy
        self.substract_pixel_mean = substract_pixel_mean


    def loadImages(self):
        f = []
        self.label_encoder= {}
        X_fishes = []
        y_fishes = []
        i=0
        o=0
        for (dirpath, dirnames, filenames) in walk(self.images_path):
            for dir_name in dirnames:
                self.label_encoder[dir_name]=i
                i+=1
                o=0
                for file in glob.glob(join(dirpath, dir_name)+"/*.png"):
                    print(file)
                    #dict_images_X[dir_name].append(file)
                    X_fishes.append(file)
                    y_fishes.append(self.label_encoder[dir_name])
                    o+=1
                    if o > self.max_items_per_class:
                        break
            break
            f.extend(filenames)
            break

        X_fishes, y_fishes = unison_shuffled_copies(np.array(X_fishes), np.array(y_fishes))
        X=[]

        for item, cat in zip(X_fishes, y_fishes):
            print(cat)
            print(item)
            X.append(cv2.imread(item))
            
        #y_fishes = to_categorical(y_fishes)
        split = int(len(X) * self.train_test_split)
        X=np.array(X)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = (X[:split], y_fishes[:split]), (X[split:], y_fishes[split:])

                
        self.num_classes = len(np.unique(y_fishes))
        self.input_shape = self.x_train.shape[1:]

        
        # normalize data.
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # if subtract pixel mean is enabled
        if self.substract_pixel_mean:
            x_train_mean = np.mean(self.x_train, axis=0)
            self.x_train -= x_train_mean
            self.x_test -= x_train_mean

        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')
        print('y_train shape:', self.y_train.shape)

        # convert class vectors to binary class matrices.
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)

    def buildNetwork(self):        
        #model transfer learning
        self.base_model = keras.applications.InceptionResNetV2( #densenet201, inceptionv3, vgg16 ...
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=self.input_shape,
            include_top=False,
            classes=self.num_classes)

        self.base_model.trainable = False
        inputs = keras.Input(shape=self.input_shape)
        # We make sure that the base_model is running in inference mode here,
        # by passing `training=False`. This is important for fine-tuning, as you will
        # learn in a few paragraphs.
        x = self.base_model(inputs, training=False)
        # Convert features of shape `base_model.output_shape[1:]` to vectors
        features_vector = keras.layers.GlobalAveragePooling2D()(x)#this flattens also
  
        self.embedding = keras.Model(inputs, features_vector)
        #image embedding model
        self.embedding.save_weights(self.model_path+"embedding_"+self.save_weights_name) #load this weights in case of an error...
        #save the model
        self.embedding.save(self.model_path+"embedding_"+self.save_model_name)

        # A Dense classifier with a single unit (binary classification)
        #outputs = keras.layers.Dense(1)(x)
        #x = Flatten()(x)
        x = keras.layers.Dropout(0.2)(features_vector)  # Regularize with dropout
        x = Dense(self.hidden_layer_size, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = Dense(self.num_classes,
                        activation='softmax')(x)
        self.model = keras.Model(inputs, outputs)


        self.model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=lr_schedule(0)),
                    metrics=['acc'])

        self.model.summary()

        try:
            self.model.load_weights(self.model_path+"nft_"+self.save_weights_name)
            print("NFT Weights loaded")

        except:
            print("NFT Weights not loaded")

    def train(self):
        # prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = '%s_model.{epoch:03d}.h5' % 'inception'
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


        history = None
        # run training, with or without data augmentation.
        if not self.data_augmentation:
            print('Not using data augmentation.')
            
            history = self.model.fit(self.x_train, self.y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_data=(self.x_test, self.y_test),
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
            datagen.fit(self.x_train)

            steps_per_epoch =  math.ceil(len(self.x_train) / self.batch_size)
            # fit the model on the batches generated by datagen.flow().
            history= self.model.fit(x=datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                    verbose=1,
                    epochs=self.epochs,
                    validation_data=(self.x_test, self.y_test),
                    steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks,
                    shuffle=True_)


        try:
            #plot history
            plt.plot(np.arange(1, self.epochs+1), history.history['loss'], label='Training Loss')
            plt.plot(np.arange(1, self.epochs+1), history.history['val_loss'], label='Validation Loss')
            plt.title('Training vs. Validation Loss', size=20)
            plt.xlabel('Epoch', size=14)
            plt.legend();
            plt.savefig('train_val_loss.png');
            plt.close()
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig('train_val_accuracy.png')
            plt.close()
        except:
            pass
        
        try:
            # score trained model
            scores = self.model.evaluate(self.x_test,
                                    self.y_test,
                                    batch_size=self.batch_size,
                                    verbose=0)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])
        except:
            print("evaluate exception")

        #nft non fine tunning
        self.model.save_weights(self.model_path+"nft_"+self.save_weights_name) #load this weights in case of an error...
        #save the model
        self.model.save(self.model_path+"nft_"+self.save_model_name)
        try:
            confusion_matrix(self.model, self.y_train, self.x_train, "conf_train.png")

        except:
            print("confusion matrices error train")
            
        try:
            confusion_matrix(self.model, self.y_test, self.x_test, "conf_test.png")

        except:
            print("confusion matrices error test")
            
    def fineTunning(self):
        #Fine tunning
        # Unfreeze the base_model. Note that it keeps running in inference mode
        # since we passed `training=False` when calling it. This means that
        # the batchnorm layers will not update their batch statistics.
        # This prevents the batchnorm layers from undoing all the training
        # we've done so far.
        self.base_model.trainable = True
        self.model.summary()


        #fine tunnning after training the classifier
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=1e-7),
                    metrics=['acc'])
        #model.compile(
        #    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        #    loss=keras.losses.BinaryCrossentropy(from_logits=True),
        #    metrics=[keras.metrics.BinaryAccuracy()],
        #)
        epochs = self.epochs_ft
        batch_size = self.batch_size_ft
        self.model.fit(self.x_train, self.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(self.x_test, self.y_test),
                        shuffle=True_)
        try:
            # score trained model
            scores = self.model.evaluate(self.x_test,
                                    self.y_test,
                                    batch_size=batch_size,
                                    verbose=0)
            print('Test loss fine tunning:', scores[0])
            print('Test accuracy:', scores[1])

        except:
            print("Error in fine tunning score part")

        self.model.save_weights(self.model_path+self.save_weights_name)
        #save the model
        self.model.save(self.model_path+self.save_model_name)
        try:
            confusion_matrix(self.model, self.y_test, self.x_test, "conf_test_finetunning.png")
            confusion_matrix(self.model, self.y_train, self.x_train, "conf_train_finetunning.png")
        
        except:
            print("fine tunnning confusion matrices error")

    def do(self):
        self.loadImages()
        self.buildNetwork()
        self.train()
        self.fineTunning()

#configs r'D:/images'
mypath = 'D:/preclassification_crop0.65' #will load folders as classes
save_model_path = 'model.h5' #from aworking directory
save_weights_path = 'weights.h5' #from working directory
model_path = ''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning Sample")
    
    parser.add_argument('-cp', '--config_path', type=str, default=r'config.yaml', help="path to the configuration file")
    parser.add_argument("-pc", "--pre_classify", action="store_true",
                        help="pre classify before object detection between target classes")
    
    parser.add_argument('-hp', '--hyper', action="store_true", help="path to the images directory with images splitted by class")
    parser.add_argument('-ip', '--images_path', type=str, default=r'D:\model10', help="path to the images directory with images splitted by class")
    parser.add_argument('-tts', '--train_test_split', type=float, default=1.0, help="path to the configuration file")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="path to the input directory")
    parser.add_argument("-bsf", "--batch_size_ft", type=int, default=3, help="path to the input directory")
    parser.add_argument("-e", "--epochs", type=int, default=250, help="path to the input model")
    parser.add_argument("-ef", "--epochs_ft", type=int, default=10, help="path to the input model")
    parser.add_argument("-mipc", "--max_items_per_class", type=int, default=310, help="path to the input weights")
    parser.add_argument("-hls", "--hidden_layer_size", type=int, default=256, help="path to the input weights")
    parser.add_argument("-da", "--data_augmentation", type=bool, default=True, help="use data augmentation?")
    parser.add_argument("-spm", "--substract_pixel_mean", type=bool, default=False, help="substract pixel mean?")
    parser.add_argument("-mp", "--model_path", type=str, default=r'inceptionresnetv2/', help="mode path with slash i.e path/")
    parser.add_argument("-smn", "--save_model_name", type=str, default=r'model.h5', help="model name")
    parser.add_argument("-swn", "--save_weights_name", type=str, default=r'weights.h5', help="weights name")
    

    args = parser.parse_args()
    print(args)

    transfer_learning = TransferLearning(args.images_path, args.train_test_split, args.batch_size, args.batch_size_ft, args.epochs, args.epochs_ft, args.max_items_per_class, args.hidden_layer_size, args.data_augmentation, args.substract_pixel_mean, args.model_path, args.save_model_name, args.save_weights_name)

    transfer_learning.do() 
