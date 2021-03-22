from sklearn.datasets import load_iris 

iris = load_iris()

dir(iris)

import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

df['target']=iris.target
df.head()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), iris.target, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)
model.score(x_test, y_test)

model = RandomForestClassifier(n_estimators=40)
model.fit(x_train, y_train)
model.score(x_test, y_test)