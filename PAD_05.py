import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import data
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split

path = 'P:/b.balcerzak/2020/INFORMATYKA_ZAOCZNE/SEM_1/PAD/PAD_05/Zadanie_domowe/trumptweets_data.csv'
#Importing file
df = pd.read_csv(path, sep=';', index_col=False, decimal=',')
#Removing two first columns which are strings
clust_df = df.drop(columns=['A', 'B'])

#Creating elbow chart to choose optimal number of cluters
inertia = []
for k in range(1,25):
    km = KMeans(n_clusters = k, random_state=0, init='k-means++')
    km.fit(clust_df)
    inertia.append([k, km.inertia_])

plt.plot(range(1,25), inertia)
plt.show()

#Chart suggests that optimal number of clusters will be 3 or 9 depending on a chart

km_clust_df = KMeans(n_clusters = 9).fit(clust_df)
clust_df['labels'] = km_clust_df.labels_

plt.pie(clust_df['labels'].value_counts(), labels=clust_df['labels'].unique())
plt.show()

km1_clust_df = KMeans(n_clusters = 3).fit(clust_df)
clust_df['labels1'] = km1_clust_df.labels_

plt.pie(clust_df['labels1'].value_counts(), labels=clust_df['labels1'].unique())
plt.show()

#looking at the pies, version with 9 clusters seems to look better 

#Calucalting cluster means for each variable

km_var_mean = clust_df.groupby(by='labels').mean()


db = DBSCAN().fit(clust_df)
clust_df['db_labels'] = db.labels_
print(clust_df['db_labels'].unique())

db_var_mean = clust_df.groupby(by='db_labels').mean()

print(km_var_mean, db_var_mean)