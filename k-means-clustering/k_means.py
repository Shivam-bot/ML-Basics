import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def euclidean_dist(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

class KMeans:

    def __init__(self,k = 5, max_iters = 100,plot_steps = False):

        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.k)]

        self.centroids = []

    def predict(self,x):

        self.x = x
        self.n_samples , self.n_features= x.shape

        random_samples_idxs = np.random.choice(self.n_samples,self.k,replace  = False)
        self.centroids = [self.x[idx] for idx  in random_samples_idxs]


        for _ in range(self.max_iters):

            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            centroid_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converge(centroid_old,self.centroids):
                break
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self,cluster):
        labels = np.empty(self.n_samples)
        for cluster_idx,cluster in enumerate(cluster):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels


    def _create_clusters(self,centroids):
        cluster = [[] for _ in range(self.k)]
        for idx,sample in enumerate(self.x):
            centroid_idx = self._closest_centroid(sample,centroids)
            cluster[centroid_idx].append(idx)
        return cluster

    def _closest_centroid(self,sample,centroids):
        distances = [euclidean_dist(sample,point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self,clusters):
        centroids = np.zeros((self.k,self.n_features))
        for cluster_idx,clusters in enumerate(clusters):
            cluster_mean = np.mean(self.x[clusters] , axis = 0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converge(self,centroid_old,centroids):
        distances = [euclidean_dist(centroid_old[i],centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def plot(self):
        fig,ax  = plt.subplots(figsize = (12,8))

        for i,index in enumerate(self.clusters):
            point = self.x[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point,marker = "x",color = "black",linewidth = 2)
        
        plt.show()


