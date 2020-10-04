import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Computing embedding")
X_red = SpectralEmbedding(n_components=2).fit_transform(X_train)
print("Done.")

clustering = SpectralClustering(n_clusters=3, assign_labels="discretize",random_state=0).fit(X_red)
y_pred = clustering.fit_predict(X_test)

print(accuracy_score(y_test, y_pred))


#Visualising the clusters of the training set

plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], s = 100, c = 'pink', label = 'Iris-virginica')

plt.show()

#Visualising the clusters
plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X_test[y_pred == 2, 0], X_test[y_pred == 2, 1], s = 100, c = 'pink', label = 'Iris-virginica')

plt.show()

