from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
data_set = pd.read_csv('sample_stocks.csv')
data_set = data_set.dropna(how='any',axis=0)

#defining the number of clusters in the method
clf = KMeans(n_clusters=3)
answer=clf.fit_predict(data_set)

#finding the silhouette score and pr
score=silhouette_score(data_set,answer)
print("silhouette score -")
print(score)

#seting the colour theme
color_theme=np.array(['red','green','yellow'])

#ploting the graph in scatter form
plt.scatter(x=data_set['returns'],y=data_set['dividendyield'],c=color_theme[clf.labels_],s=100)
plt.show()
