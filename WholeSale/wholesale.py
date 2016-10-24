import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.cluster import KMeans



URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
response = requests.get(URL)
outpath  = os.path.abspath("customers.txt")
with open(outpath, 'wb') as f:
    f.write(response.content)
df = pd.read_csv("customers.txt", usecols=["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"])

data = df.values
model = KMeans(n_clusters = 7)
model.fit(data)

labels = model.labels_
centroids = model.cluster_centers_

for i in range(7):
	datapoints = data[np.where(labels==i)]
	plt.plot(datapoints[:,3],datapoints[:,4],'k.')
	centers = plt.plot(centroids[i,3],centroids[i,4],'x')
	plt.setp(centers,markersize=20.0)
	plt.setp(centers,markeredgewidth=5.0)

plt.xlim([0,10000])
plt.ylim([0,15000])
savefig('kClusterWholesale.png')
plt.show()