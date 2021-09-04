import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from random_forest import RandomForest


def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)


data = datasets.load_breast_cancer()

x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 123)
clf = RandomForest(n_trees = 3,max_depth = 10)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
acc = accuracy(y_test,y_pred)

print("Accuracy:",acc)