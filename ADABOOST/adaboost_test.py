import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from adaboost import Adaboost

def accuracy(y_true,y_pred):
    return np.sum(y_true == y_pred)/len(y_true)

data = datasets.load_breast_cancer()
x= data.data
y = data.target

y[y==0] = -1
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=9)

clf = Adaboost(n_clf = 5)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = accuracy(y_test,y_pred)
print(f"Accuracy  is {acc}")