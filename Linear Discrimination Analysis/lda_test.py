from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from lda import LDA

data = datasets.load_iris()

x = data.data
y = data.target

lDa = LDA(3)
lDa.fit(x,y)

x_projected = lDa.transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

plt.scatter(x1,x1,c=y,edgecolor = 'none',alpha  = 0.8,cmap = plt.cm.get_cmap('viridis',3))

plt.xlabel("LINEAR DISCRIMANT 1")
plt.ylabel("LINEAR DISCRIMANT 2")
plt.colorbar()
plt.show()


