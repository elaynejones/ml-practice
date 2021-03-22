import pandas as pd

df=pd.read_csv('homeprices.csv')
df

#Pandas for Dummy Variables

dummies = pd.get_dummies(df.town)
dummies

merged_df = pd.concat([df, dummies], axis='columns')
merged_df

final_df = merged.drop(['town'], axis='columns')
final_df

#Accounting for the dummy variable trap of multi-colinearity 

final = final_df.drop(['west windsor'], axis='columns')
final

X= final.drop('price', axis='columns')
X

Y=final.price

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(X, Y)

model.predict(X)

model.score(X,Y)


#Using OneHotEncoder

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

df_label = df 

df_label.town = label.fit_transform(df_label.town)
df_label

#double brackets give a list
X=df_label[['town', 'area']].values
X

Y=df_label.price.values
Y

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

column = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')

X=column.fit_transform(X)
X

X= X[:, 1:]

X

model.fit(X,Y)

