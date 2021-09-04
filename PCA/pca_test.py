from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from pca import PCA

data = datasets.load_iris()

x = data.data
y = data.target

pca = PCA(2)
pca.fit(x)
x_projected = pca.transform(x)

print('Shape of x:',x.shape)
print('Shape of x transform:',x_projected.shape)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

plt.scatter(x1,x2,c = y,edgecolor='none',alpha=0.8,cmap=plt.cm.get_cmap('viridis',3))

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar()
plt.show()
