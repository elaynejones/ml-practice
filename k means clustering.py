import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()

df['flower'] = iris.target
df.head()

df.drop(['sepal length (cm)', 'sepal width (cm)', 'flower'], axis='columns', inplace=True)
df.head()

km = KMeans(n_clusters =3)
y_pred = km.fit_predict(df)
y_pred

df['cluster'] = y_pred
df.head

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='red')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='yellow')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='green', marker='+', label='centroid')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()


#Making the Elbow Plot to find best number of K

sse = [] 
k_range = range(1,10)
for k in k_range:
    km = KMeans(n_clusters = k)
    km.fit(df)
    sse.append(km.intertia_)

plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
plt.plot(k_range, sse)