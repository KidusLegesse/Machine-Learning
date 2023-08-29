import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf

cols = ["age", "sex", "chestpain", "trestbps", "chol", "fbs",
        "restecg", "maxHr", "exang", "oldpeak", "slope", "ca", "thal", "Hd"]
df = pd.read_csv("processed.cleveland.data", names=cols)
train, valid, test = np.split(
    df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y


train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=5)
y_valid_one_hot = tf.keras.utils.to_categorical(y_valid, num_classes=5)

