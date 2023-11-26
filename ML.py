import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from sklearn.impute import SimpleImputer

#Each column is an attribute
attributes = ["age", "sex", "chestpain", "trestbps", "chol", "fbs",
        "restecg", "maxHr", "exang", "oldpeak", "slope", "ca", "thal", "Hd"]
dataSet = pd.read_csv("processed.cleveland.data", names=attributes)

#Checking to see if the the columns are properly named and placed
print(dataSet.head())

#Replacing missing values with NaN since they are represented by ?
dataSet.replace('?', np.nan, inplace=True)

#Replacing missing values(which are (?)) with mean of the column
replace_missing= SimpleImputer(strategy='mean')
clean_dataset = pd.DataFrame(replace_missing.fit_transform(dataSet), columns=dataSet.columns)

#Checking to see if my data still has any missing values
print(clean_dataset.isnull().sum()) 

#Splitting the data into three different sets
training_data, validation_data, testing_data = np.split(
    clean_dataset.sample(frac=1), [int(0.6*len(clean_dataset)), int(0.8*len(clean_dataset))])

#Checking the amount of data allocated to each dataset
print(len(training_data))
print(len(validation_data))
print(len(testing_data))

def scale_dataset(dataframe, oversample=False):
    df_attributes = dataframe[dataframe.columns[:-1]].values
    old_targets = dataframe[dataframe.columns[-1]].values
    #Converted targets to binary by changing changing all 2,3,4 values into 1. (Explanation in Readme)
    new_targets = np.where(old_targets != 0, 1, old_targets)

    scaler = StandardScaler()
    df_attributes = scaler.fit_transform(df_attributes)

    if oversample:
        df_attributes, new_targets = RandomOverSampler().fit_resample(df_attributes, new_targets)

    new_data = np.hstack((df_attributes, np.reshape(new_targets, (-1, 1))))

    return new_data, df_attributes, new_targets


train, attributes_train, targets_train = scale_dataset(training_data, oversample=True)
valid, attributes_valid, targets_valid = scale_dataset(validation_data, oversample=False)
test, attributes_test, targets_test = scale_dataset(testing_data, oversample=False)

targets_train_one_hot = tf.keras.utils.to_categorical(targets_train, num_classes=2)
targets_test_one_hot = tf.keras.utils.to_categorical(targets_test, num_classes=2)
targets_valid_one_hot = tf.keras.utils.to_categorical(targets_valid, num_classes=2)
