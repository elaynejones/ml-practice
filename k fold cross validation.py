from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


iris = load_iris()

#could also do StratifiedKFold - which divides the classes in uniform way
kfold = KFold(n_splits=10)
kfold

#this is an example of what cross val score is doing behind the scenes:
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

logmodel = cross_val_score(LogisticRegression(), iris.data, iris.target)
logmodel

np.average(logmodel)


dectree = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target)
dectree

np.average(dectree)


randomforest = cross_val_score(RandomForestClassifier(), iris.data, iris.target)
randomforest

np.average(randomforest)


svcmodel = cross_val_score(SVC(), iris.data, iris.target)
svcmodel

np.average(svcmodel))