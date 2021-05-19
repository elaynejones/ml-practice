import pandas as pd

data = pd.read_csv('data/clusters.csv')
data.head()

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

scaled_features= MinMaxScaler().fit_transform(data)
pca=PCA(n_components=2).fit(scaled_features)
features_2d=pca.transform(scaled_features)
features_2d[0:10]

import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(features_2d[:, 0}, features_2d[:,1]])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data')
plt.show() 

import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans
%matplotplib inline 

wcss = [] 
for i in range(1, 11):
    kmeans=KMeans(n_clusters = i)
    kmeans.fit(features.values)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('WCSS by Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


model = KMeans(n_clusters=4, init='k-means++', n_init=100, max_inter=1000)
km_clusters=model.fit_predict(features.values)
km_clusters

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange', 3:'cyan'}
    mrk_dic = {0:'*',1:'x',2:'+', 3:'.'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)



#trying agglomerative

from sklearn.cluster import AgglomerativeClustering

agg_model = AgglomerativeClustering(n_clusters=4)
agg_clusters = agg_model.fit_predict(features.values)
agg_clusters

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange', 3:'cyan'}
    mrk_dic = {0:'*',1:'x',2:'+', 3:'.'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()