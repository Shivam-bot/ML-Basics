import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from k_means import KMeans

x,y = make_blobs(centers = 4, n_samples = 500, n_features=2,shuffle = True,random_state=123)
print(x.shape)

clusters = len(np.unique(y))
print(clusters)
k = KMeans(k = clusters,max_iters = 150,plot_steps = False)

y_pred = k.predict(x)

k.plot()