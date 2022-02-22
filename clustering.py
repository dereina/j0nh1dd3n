import argparse
import os
import sys
import glob
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import shutil

print(__file__)
print(os.getcwd())
print(sys.argv[0])
print(os.path.dirname(sys.argv[0]))
os.chdir(os.path.dirname(sys.argv[0]))
class Clustering():
    def __init__(self, input, output, model_path, weights_path):
        self.input = input
        self.output = output
        self.model_path = model_path
        self.weights_path = weights_path
    
    def loadImages(self):
        self.X = []
        self.files = []
        #i=100
        for file in glob.glob(self.input+"/*.png"):
            self.files.append(file)
            self.X.append(cv2.imread(file))
            #i-=1
            #if i == 0:
            #    break

        self.X = np.array(self.X)
        pass
    
    def loadModel(self):
        self.embedding_model = load_model(self.model_path)
        self.embedding_model.load_weights(self.weights_path)

    def cluster(self):
        self.embeddings = []
        for i in range(len(self.X)):
            aux = np.array([self.X[i]])
            embedding = self.embedding_model(aux)
            self.embeddings.append(embedding[0])

        self.embeddings = np.array(self.embeddings)
        print(self.embeddings.shape)
        print(self.embeddings[0])
        #haz el clustering, kmeans o louvain o hierarchical...
        kmeans = KMeans(
            init="random",
            n_clusters=11,
            n_init=10,
            max_iter=300,
            random_state=42
        )
        kmeans.fit(self.embeddings)
        print("labels")
        print(kmeans.labels_)
        print("centers")
        print(kmeans.cluster_centers_)
        for i in range(len(kmeans.cluster_centers_)):
            try:
                os.mkdir(self.output +"/"+str(i), 0x755 );

            except:
                pass
       
        for i in range(len(kmeans.labels_)):
            shutil.copy(self.files[i], self.output +"/"+str(kmeans.labels_[i])) 

        

if __name__ == "__main__":
    default_inou = r'D:\clustering'
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-o', '--output', type=str, default=default_inou, help="path to the output directory")
    parser.add_argument('-cp', '--config_path', type=str, default=r'config.yaml', help="path to the configuration file")
    parser.add_argument("-i", "--input", type=str, default=default_inou, help="path to the input directory")
    parser.add_argument("-m", "--model_path", type=str, default=r'inceptionresnetv2/embedding_model.h5', help="path to the input model")
    parser.add_argument("-w", "--weights_path", type=str, default=r'inceptionresnetv2/embedding_weights.h5', help="path to the input weights")
    args = parser.parse_args()
    print(args)
    clustering = Clustering(args.input, args.output, args.model_path, args.weights_path)
    clustering.loadImages()
    clustering.loadModel()
    clustering.cluster()
    #for root, dirs, files in os.walk(args.input, topdown=True):
    #    for i, dir in enumerate(dirs):
    #        input = join(root, dir)
    #        inference = Inference(input, input, args.model, args.weights, args.list_classes, args.prepend_probability, args.threshold, args.threshold_alt, args.threshold_mig, args.crop)
    #        inference.infiere()
    #    break