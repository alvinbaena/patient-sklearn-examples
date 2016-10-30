'''
Created on 6/06/2016
Visualiza el dataset de pacientes
@author: Andres Moreno B.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

how_many_visualize = 2000
np.random.seed(0)

patient_data = pd.read_pickle("../../data/df/dataset.pickle").values
indices = np.random.permutation(len(patient_data))
target = pd.read_pickle("../../data/df/target.pickle").values

centers = [[1, 1], [-1, -1], [1, -1]]
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# keep 80% of variance
pca = decomposition.PCA(n_components=0.95)
pca.fit(patient_data[indices[1:how_many_visualize]])
X = pca.transform(patient_data[indices[1:how_many_visualize]])
target = target[[indices[1:how_many_visualize]]]

# for name, label in [('Dead', 0), ('Survive', 1)]:
#    ax.text3D(X[target == label, 0].mean(),
#              X[target == label, 1].mean() + 1.5,
#              X[target == label, 2].mean(), name,
#              horizontalalignment='center',
#              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# # Reorder the labels to have colors matching the cluster results
# y = np.choose(target, [1, 2, 0]).astype(np.float)

print target
print type(target)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=target)

# x_surf = [X[:, 0].min(), X[:, 0].max(),
#           X[:, 0].min(), X[:, 0].max()]
# y_surf = [X[:, 0].max(), X[:, 0].max(),
#           X[:, 0].min(), X[:, 0].min()]
# x_surf = np.array(x_surf)
# y_surf = np.array(y_surf)
# v0 = pca.transform(pca.components_[[0]])
# v0 /= v0[-1]
# v1 = pca.transform(pca.components_[[1]])
# v1 /= v1[-1]
#
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])

plt.show()
