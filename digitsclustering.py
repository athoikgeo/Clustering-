print(__doc__)
from time import time
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.metrics import accuracy_score
from sklearn.cluster import SpectralClustering


X, y = datasets.load_digits(return_X_y=True)
n_samples, n_features = X.shape

np.random.seed(0)

def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                  .3 * np.random.normal(size=2),
                                  mode='constant',
                                  ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


X, y = nudge_images(X, y)


# Visualize the clustering
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")



for affinity in ('rbf', 'laplacian','nearest_neighbors'):
    clustering = SpectralClustering(affinity=affinity, n_clusters=10)
    t0 = time()
    clustering.fit(X_red[:3000])
    print("%s :\t%.2fs" % (affinity, time() - t0))

    plot_clustering(X_red[:3000], clustering.labels_, "%s affinity" % affinity)


    plt.show()

for affinity in ('rbf', 'laplacian','nearest_neighbors'):
    clustering = SpectralClustering(affinity=affinity, n_clusters=10)
    t0 = time()
    clustering.fit(X_red[:3000],y[:3000])
    print("%s :\t%.2fs" % (affinity, time() - t0))
    y_pred = clustering.fit_predict(X_red[3000:])
    print(accuracy_score(y[3000:], y_pred))
    plot_clustering(X_red[3000:], clustering.labels_, "%s affinity" % affinity)
    plt.show()



