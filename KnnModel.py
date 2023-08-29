import numpy as np
import ML
from sklearn.metrics import classification_report


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KNN:
    def __init__(self, neighbors=3):
        self.neighbors = neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self.predictHelper(x) for x in X]
        return np.array(y_pred)

    def predictHelper(self, x):
        distances = [euclidean_distance(x, x_train)
                     for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)


knn_model = KNN(5)
knn_model.fit(ML.X_train, ML.y_train)
y_pred = knn_model.predict(ML.X_test)
print(knn_model.accuracy(ML.y_test, y_pred))
print(classification_report(ML.y_test, y_pred, zero_division=1))
