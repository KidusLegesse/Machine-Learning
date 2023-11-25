from sklearn.metrics import classification_report
import numpy as np
import ML as ML


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

class CustomKNNModel:
    def __init__(self, neighbors=3):
        self.neighbors = neighbors
        self.attributes = None
        self.targets = None
    
    def fit(self, attributes, targets):
        self.attributes = attributes
        self.targets = targets

    # Returns an array of predictions by calculating the distance of all training points
    def predict(self, test_data, lst=[]):
        
        for data in test_data:
            distances = [euclidean_distance(data, attribute) for attribute in self.attributes]
            neighbors_index = np.argsort(distances)[:self.neighbors]
            kneighbors= [self.targets[point] for point in neighbors_index]

            # Predict the class based on majority vote
            target_prediction = max(set(kneighbors), key=kneighbors.count)
            lst.append(target_prediction)

        return np.array(lst)

# Training the model using the training set
knn_model = CustomKNNModel(neighbors=3)
knn_model.fit(ML.attributes_train, ML.targets_train)

# Chekcing the accuracy of the model on the validation set
valid_target_pred = knn_model.predict(ML.attributes_valid)
accuracy = np.mean(valid_target_pred == ML.targets_valid)
accuracy = (round(accuracy*100)) # round and convert accuracy to a percentage for readability
print("Validation Accuracy:", str(accuracy)+"%") 

# Classification report on using the test set
target_pred = knn_model.predict(ML.attributes_test, [])
print(classification_report(ML.targets_test, target_pred))
