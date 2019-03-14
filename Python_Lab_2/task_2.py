from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('stocks_sample_data.csv')

#to a varibale called Squarred_error_sum to append all values of KMeans Cluster
Squarred_err_sum=[]

#setting the range
K=range(1,20)
for k in K:
    clf = KMeans(n_clusters=k)
    clf.fit(data)
    Squarred_err_sum.append(clf.inertia_)

#plotting in the graph
plt.plot(K,Squarred_err_sum,'bx-')
#naming the X coordinates
plt.xlabel("K_Values")
#naming the Y coordinate
plt.ylabel("squared_error_sum")

plt.show()

