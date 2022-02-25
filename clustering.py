import argparse
import os
import shutil
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS

from sklearn import metrics
import matplotlib.pyplot as plt

print(__file__)
print(os.getcwd())
print(sys.argv[0])
print(os.path.dirname(sys.argv[0]))
os.chdir(os.path.dirname(sys.argv[0]))
class Clustering():
    def __init__(self, input, output, model_path, weights_path, clustering, nclusters, min_samples):
        self.input = input
        self.output = output
        self.model_path = model_path
        self.weights_path = weights_path
        self.clustering = clustering
        self.nclusters = nclusters
        self.min_samples = min_samples
    
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

    def computeEmbeddings(self):
        self.embeddings = []
        for i in range(len(self.X)):
            aux = np.array([self.X[i]])
            embedding = self.embedding_model(aux)
            self.embeddings.append(embedding[0])
            #if i == 100:
            #    break

        self.embeddings = np.array(self.embeddings)
        return self.embeddings

    def saveResults(self, num_clusters, labels, minus_one = 0):
        for i in range(num_clusters + minus_one):
            try:
                os.mkdir(self.output +"/"+str(i), 0x755 );

            except:
                pass
        try:
            os.mkdir(self.output +"/"+str(minus_one), 0x755 );

        except:
            pass
       
        for i in range(len(labels)):
            shutil.copy(self.files[i], self.output +"/"+str(labels[i])) 


    def cluster(self):
        if self.clustering == 0:
            self.kmeans()
        
        elif self.clustering == 1:
            self.dbscan()
        
        elif self.clustering == 2:
            self.aglomerative()
        
        elif self.clustering == 3:
            self.spectral()

        elif self.clustering == 4:
            self.birch()

        elif self.clustering == 5:
            self.affinityPropagation()
        
        elif self.clustering == 6:
            self.optics()

    def kmeans(self):
        self.computeEmbeddings()
        print(self.embeddings.shape)
        print(self.embeddings[0])
        #haz el clustering, kmeans o louvain o hierarchical...
        kmeans = KMeans(
            init="random",
            n_clusters=self.nclusters,
            n_init=10,
            max_iter=10000,
            random_state=42
        )
        kmeans.fit(self.embeddings)
        print("labels")
        print(kmeans.labels_)
        print("centers")
        print(kmeans.cluster_centers_)
        self.saveResults(len(kmeans.cluster_centers_), kmeans.labels_)

    def dbscan(self):
        self.computeEmbeddings()
        X = StandardScaler().fit_transform(self.embeddings)

        # #############################################################################
        # Compute DBSCAN
        db = DBSCAN(eps=0.3, min_samples=self.min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        #print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        #print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        #print(
        #    "Adjusted Mutual Information: %0.3f"
        #    % metrics.adjusted_mutual_info_score(labels_true, labels)
        #)
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
    
        self.saveResults(len(set(db.labels_)), db.labels_, -1)


    def aglomerative(self):
        self.computeEmbeddings()
        cluster = AgglomerativeClustering(n_clusters=self.nclusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(self.embeddings)

        print(set(cluster.labels_))
        print(cluster.labels_)
        plt.scatter(self.embeddings[:,0],self.embeddings[:,1], c=cluster.labels_, cmap='rainbow')
        plt.show()
        self.saveResults(len(set(cluster.labels_)), cluster.labels_)

    def spectral(self):
        self.computeEmbeddings()
        clustering = SpectralClustering(n_clusters=self.nclusters,
        assign_labels='discretize',
        random_state=0).fit(self.embeddings)
        self.saveResults(len(set(clustering.labels_)), clustering.labels_)

    def birch(self):
        self.computeEmbeddings()
        cluster = Birch(n_clusters=self.nclusters)
        cluster.fit(self.embeddings)
        cluster.predict(self.embeddings)

        print(set(cluster.labels_))
        print(cluster.labels_)
        plt.scatter(self.embeddings[:,0],self.embeddings[:,1], c=cluster.labels_, cmap='rainbow')
        plt.show()
        self.saveResults(len(set(cluster.labels_)), cluster.labels_)
    
    def affinityPropagation(self):
        self.computeEmbeddings()
        clustering = AffinityPropagation(max_iter = 1000, convergence_iter=100,random_state=5)
        clustering.fit(self.embeddings)
        clustering.predict(self.embeddings)
        print(set(clustering.labels_))
        print(clustering.labels_)
        self.saveResults(len(set(clustering.labels_)), clustering.labels_, -1)

    def optics(self):
        self.computeEmbeddings()
        clustering = OPTICS(min_samples=self.min_samples)
        clustering.fit_predict(self.embeddings)
        print(set(clustering.labels_))
        print(clustering.labels_)
        self.saveResults(len(set(clustering.labels_)), clustering.labels_, -1)

if __name__ == "__main__":
    default_inou = r'D:\clustering'
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-o', '--output', type=str, default=default_inou, help="path to the output directory")
    parser.add_argument('-cp', '--config_path', type=str, default=r'config.yaml', help="path to the configuration file")
    parser.add_argument("-i", "--input", type=str, default=default_inou, help="path to the input directory")
    parser.add_argument("-m", "--model_path", type=str, default=r'inceptionresnetv2/embedding_model.h5', help="path to the input model")
    parser.add_argument("-w", "--weights_path", type=str, default=r'inceptionresnetv2/embedding_weights.h5', help="path to the input weights")
    parser.add_argument("-c", "--clustering", type=int, default=5, help="0-kmeans 1-dbscan 2-aglomerative 3-spectral 4-birch 5-affiinity propagation 6-optics")
    parser.add_argument("-n", "--nclusters", type=int, default=7, help="number of clusters")
    parser.add_argument("-ms", "--min_samples", type=int, default=10, help="min samples in the neighbourhood")


    args = parser.parse_args()
    print(args)
    clustering = Clustering(args.input, args.output, args.model_path, args.weights_path, args.clustering, args.nclusters, args.min_samples)
    clustering.loadImages()
    clustering.loadModel()
    clustering.cluster()
    #for root, dirs, files in os.walk(args.input, topdown=True):
    #    for i, dir in enumerate(dirs):
    #        input = join(root, dir)
    #        inference = Inference(input, input, args.model, args.weights, args.list_classes, args.prepend_probability, args.threshold, args.threshold_alt, args.threshold_mig, args.crop)
    #        inference.infiere()
    #    break
