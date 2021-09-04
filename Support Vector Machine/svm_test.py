import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from svm import SVM


x,y = datasets.make_blobs(n_samples=50,n_features=2,centers = 2,cluster_std=1.05,random_state=123)
y = np.where(y == 0,1,-1)

clf = SVM()
clf.fit(x,y)
predictions = clf.predict(x)
print(predictions)