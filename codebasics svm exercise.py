import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

dir(digits)

digits.target

digits.target_names

df = pd.DataFrame(digits.data, digits.target)
df.head()

df['target']=digits.target
df.head()

from sklearn.model_selection import train_test_split

X= df.drop(['target', 'target_name'], axis='columns')
Y = df.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#RBF Kernel

from sklearn.svm import SVC

rbfmodel= SVC(kernel='rbf')

rbfmodel.fit(x_train, y_train)

rbfmodel.score(x_test, y_test)


#Linear Kernel

from sklearn.svm import SVC

linearmodel = SVC(kernel='linear')

linearmodel.fit(x_train, y_train)

linearmodel.score(x_test, y_test)


#Regularization

cmodel = SVC(C=1)
cmodel.fit(x_train, y_train)
cmodel.score(x_test, y_test)

cmodel = SVC(C=10)
cmodel.fit(x_train, y_train)
cmodel.score(x_test, y_test)


#Gamma 

gmodel = SVC(gamma=1)
gmodel.fit(x_train, y_train)
gmodel.score(x_test, y_test)

gmodel = SVC(gamma=10)
gmodel.fit(x_train, y_train)
gmodel.score(x_test, y_test)
