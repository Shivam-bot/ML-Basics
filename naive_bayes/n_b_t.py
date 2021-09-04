import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from naive_byes import NaiveBayes

print("Code is Executing")
def accuracy(y_true,y_predict):
    accuracy_ = np.sum(y_true==y_predict)/len(y_true)
    return accuracy_

x,y = datasets.make_classification(n_samples = 1000,n_features = 10,n_classes = 2,random_state = 123)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 123)

nb = NaiveBayes()
nb.fit(x_train,y_train)
predictions = nb.predict(x_test)

print("accuracy is ",accuracy(y_test,predictions))