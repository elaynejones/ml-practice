from sklearn import datasets
wine=datasets.load_wine()

dir(wine)

wine.feature_names

wine.target_names

import pandas as pd

df = pd.DataFrame(wine.data, columns=feature_names)
df.head()

df['target']=wine.target
df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=100)

from sklearn.naive_bayes import GaussianNB, MultinomialNB

gaussian= GaussianNB()
gaussian.fit(X_train, y_train)


gaussian.score(X_test, y_test)

multi = MultinomialNB()
multi.fit(X_train, y_train)

multi.score(X_test, y_test)