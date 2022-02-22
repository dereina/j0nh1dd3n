import argparse
from email.policy import default
import enum
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

import shutil
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
    def __init__(self, input, output, model, weights, list_classes, prepend_probability, threshold=0.3333, threshold_alt=0.3333, threshold_mig=0.3333, crop_image = False):
        self.input = input
        self.output = output
        self.model = model
        self.weights = weights
        self.int_to_string = {}
        for i, c in enumerate(list_classes):
            self.int_to_string[i] = c
        
        self.prepend_probability = prepend_probability
        self.threshold = threshold
        self.threshold_alt = threshold_alt
        self.threshold_mig = threshold_mig
        self.crop_image = crop_image
        self.classifier_model = load_model(self.model)
        self.classifier_model.load_weights(self.weights)
        
    def crop(self, im):
        print(im.shape)
        size = im.shape[1] // 3
        sl = im[::,size:(im.shape[1]-size)]
        #cv2.imshow(sl)
        return sl

    def infiere(self):
        #path_non = self.output + "/non"
        path_t = self.output + "/threshold_rebuig"
        path_ta = self.output + "/threshold_alt"
        path_tm = self.output + "/threshold_mig"
        path_tb = self.output + "/threshold_baix"
        #try:
        #    os.mkdir(path_non, 0x755 );
        
        #except:
        #    pass
        try:
            shutil.rmtree(path_t);
            shutil.rmtree(path_ta);
            shutil.rmtree(path_tm);
            shutil.rmtree(path_tb);
        
        except:
            pass

        try:
            os.mkdir(path_t, 0x755 );
            os.mkdir(path_ta, 0x755 );
            os.mkdir(path_tm, 0x755 );
            os.mkdir(path_tb, 0x755 );
        
        except:
            pass

        for root, dirs, files in os.walk(self.input, topdown=False):
            for name in files:
                if name == "Thumbs.db":
                    continue
                
                print(os.path.join(root, name))
                ia = cv2.imread(os.path.join(root, name))
                print(ia.shape)
                ia = cv2.cvtColor(ia, cv2.COLOR_RGBA2RGB)
                if self.crop_image:
                    ia = self.crop(ia)

                #do inference classifier here
                to_predict = np.array([ia])
                to_predict = to_predict.astype('float32') / 255 #input normalization or network won't work...

                out_y = np.array([[1,0,0,0,0,0]]) #if pre_Classify is False we want to do the object detection anyway
                out_y = self.classifier_model.predict(to_predict)
                index = np.argmax(out_y[0])
                print(index)
                path = self.output + "/"
                if self.threshold <= out_y[0][index]:
                    path = path_t + "/"
                
                elif self.threshold_alt <= out_y[0][index]:
                    path = path_ta + "/"
                
                elif self.threshold_mig <= out_y[0][index]:
                    path = path_tm + "/"
                
                else:
                    path = path_tb + "/"
                
                try:
                    os.mkdir(path, 0x755 );
                
                except:
                    pass
                
                path += self.int_to_string[index]
                try:
                    os.mkdir(path, 0x755 );
                
                except:
                    pass

                
                
                #if self.threshold <= out_y[0][index]:
                if self.prepend_probability:
                    cv2.imwrite(path + "/"+str(out_y[0][index])+"_"+name, ia)
                
                else:
                    cv2.imwrite(path + "/"+name, ia)
                
                #else:
                    #cv2.imwrite(path_non + "/"+int_to_string[index]+"_"+str(out_y[0][index])+"_"+name, ia)
                    #cv2.imwrite(path_non + "/"+name, ia)

            for name in dirs:
                print(os.path.join(root, name))
                
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # Optional argument
    default_inou = r'D:\model10'
    parser.add_argument('-o', '--output', type=str, default=default_inou, help="path to the output directory")
    parser.add_argument('-cp', '--config_path', type=str, default=r'config.yaml', help="path to the configuration file")
    parser.add_argument("-i", "--input", type=str, default=default_inou, help="path to the input directory")
    parser.add_argument("-m", "--model", type=str, default=r'inceptionresnetv2/model.h5', help="path to the input model")
    parser.add_argument("-w", "--weights", type=str, default=r'inceptionresnetv2/weights.h5', help="path to the input weights")
    parser.add_argument("-t", "--threshold", type=float, default=0.4, help="probability threshold")
    parser.add_argument("-ta", "--threshold_alt", type=float, default=0.7, help="probability threshold alt")
    parser.add_argument("-tm", "--threshold_mig", type=float, default=0.5, help="probability threshold mig")

    parser.add_argument("-c", "--crop", type=bool, default=False, help="crop image width by 1/3")
    parser.add_argument("-pp", "--prepend_probability", type=bool, default=False, help="prepends the probability to image name")
    parser.add_argument("-lc", "--list_classes", nargs="+", default=["back", "empty", "lomo_abierto",  "manchas","rosa", "sano"])

    args = parser.parse_args()
    print(args)
    for root, dirs, files in os.walk(args.input, topdown=True):
        for i, dir in enumerate(dirs):
            input = join(root, dir)
            inference = Inference(input, input, args.model, args.weights, args.list_classes, args.prepend_probability, args.threshold, args.threshold_alt, args.threshold_mig, args.crop)
            inference.infiere()
        break
    
