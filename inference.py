import argparse
from signal import signal, SIGTERM
from concurrent import futures
import random

import grpc
import cv2
import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import time

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from tensorflow.keras.models import load_model
from os import walk
from os.path import join
import yaml
tf.get_logger().setLevel('ERROR')  

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(__file__)
print(os.getcwd())
print(sys.argv[0])
print(os.path.dirname(sys.argv[0]))
print(tf.__version__)


class Inference:
    def __init__(self, input, output, model, weights, threshold=0.3333):
        self.input = input
        self.output = output
        self.model = model
        self.weights = weights
        self.threshold = threshold
        self.classifier_model = load_model(self.model)
        self.classifier_model.load_weights(self.weights)

    def crop(self, im):
        print(im.shape)
        size = im.shape[1] // 3
        sl = im[::,size:(im.shape[1]-size)]
        #cv2.imshow(sl)
        return sl

    def infiere(self):
        path_non = self.output + "/non"
        try:
            os.mkdir(path_non, 0x755 );
        
        except:
            pass

        for root, dirs, files in os.walk(self.input, topdown=False):
            for name in files:
                print(os.path.join(root, name))
                ia = cv2.imread(os.path.join(root, name))
                print(ia.shape)
                ia = cv2.cvtColor(ia, cv2.COLOR_RGBA2RGB)

                #do inference classifier here
                to_predict = np.array([ia])
                to_predict = to_predict.astype('float32') / 255 #input normalization or network won't work...

                out_y = np.array([[1,0,0,0,0,0]]) #if pre_Classify is False we want to do the object detection anyway
                out_y = self.classifier_model.predict(to_predict)
                index = np.argmax(out_y[0])
                print(index)
                path = self.output + "/"+str(index)
                try:
                    os.mkdir(path, 0x755 );
                
                except:
                    pass
                
                ia = self.crop(ia)
                if self.threshold <= out_y[0][index]:
                    cv2.imwrite(path + "/"+name, ia)
                
                else:
                    cv2.imwrite(path_non + "/"+name, ia)

            for name in dirs:
                print(os.path.join(root, name))
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # Optional argument
    parser.add_argument('-o', '--output', type=str, default=r'D:/images', help="path to the output directory")
    parser.add_argument('-cp', '--config_path', type=str, default=r'config.yaml', help="path to the configuration file")
    parser.add_argument("-i", "--input", type=str, default=r'D:/images', help="path to the input directory")
    parser.add_argument("-m", "--model", type=str, default=r'inceptionresnetv2/model.h5', help="path to the input model")
    parser.add_argument("-w", "--weights", type=str, default=r'inceptionresnetv2/weights.h5', help="path to the input weights")
    parser.add_argument("-t", "--threshold", type=float, default=0.65, help="probability threshold")
    
    args = parser.parse_args()
    print(args)

    inference = Inference(args.input, args.output, args.model, args.weights, args.threshold)
    inference.infiere()
    #print(args.config_path)
    #serve(args)