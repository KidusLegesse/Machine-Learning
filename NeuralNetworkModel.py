import tensorflow as tf
import ML
import matplotlib.pyplot as plt


#Function for training the Neural Network model
def train_NeuralNetwork(att_train, tar_train):
    """"
    Explaining params:
    -Dropout layer is 0.3 to prevent overfitting
    -The batch size of 32 gave the best accuracy.
    -I chose the loss to be binary cross entropy since there are 2 targets [0,1]
    -Relu function is used in the hidden layers because of its effiecnt
    -Sigmoid function for the output layers because this is a binay classification
    - Nodes (64, 32, 32) gave the best accuracy
    """
    neuralNet = tf.keras.Sequential()
    neuralNet.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)))
    neuralNet.add(tf.keras.layers.Dropout(0.3))
    neuralNet.add(tf.keras.layers.Dense(32, activation='relu',))
    neuralNet.add(tf.keras.layers.Dropout(0.3))
    neuralNet.add(tf.keras.layers.Dense(32, activation='relu'))
    neuralNet.add(tf.keras.layers.Dropout(0.3))
    neuralNet.add(tf.keras.layers.Dense(2, activation='sigmoid'))

    neuralNet.compile(optimizer="adam", loss='binary_crossentropy',
                  metrics=['accuracy'])
    model_history = neuralNet.fit(
        att_train, tar_train, epochs=64, batch_size=32, validation_data=(ML.attributes_valid, ML.targets_valid_one_hot), verbose=0
    )
    return neuralNet, model_history

#Plots the history of the model.
def model_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    #The first plot compares the loss of the VValidational and Training
    ax1.plot(history.history['loss'], label='Training-Data', color='#e63946')
    ax1.plot(history.history['val_loss'], label='Validation-Data', color='#f1faee')
    ax1.set_xlabel('Epoch',color='#a8dadc')
    ax1.set_ylabel('Loss', color='#a8dadc')
    ax1.set_title('NeuralNet-Loss', color='#a8dadc')
    ax1.grid(True)
    legend=ax1.legend()
    legend.set_frame_on(True)
    legend.get_frame().set_facecolor('#a8dadc')
    ax1.set_facecolor('#1d3557')
    ax1.tick_params(axis='x', colors="#e63946")
    ax1.tick_params(axis='y', colors="#f1faee")
    
    #The second plot compares the accuracy of the Validation and Training
    ax2.plot(history.history['accuracy'], label='Training-Data', color='#1d3557')
    ax2.plot(history.history['val_accuracy'], label='Validation-Data',  color='#e63946')
    ax2.set_xlabel('Epoch', color='#a8dadc')
    ax2.set_ylabel('Accuracy', color='#a8dadc')
    ax2.set_title('NeuralNet-Accuracy', color='#a8dadc')
    ax2.grid(True)
    legend=ax2.legend()
    legend.set_frame_on(True)
    legend.get_frame().set_facecolor('#a8dadc')
    ax2.set_facecolor('#f1faee')
    ax2.tick_params(axis='x', colors="#f1faee")
    ax2.tick_params(axis='y', colors="#e63946")

    fig.set_facecolor('#272829')

    plt.show()

#Creating the model along with plotting its history
neuralNet_model, history = train_NeuralNetwork(ML.attributes_train, ML.targets_train_one_hot)
model_history(history)

#Evaluating the preformance of the model.
val_loss, val_accuracy = neuralNet_model.evaluate(ML.attributes_valid, ML.targets_valid_one_hot)
test_loss, test_accuracy = neuralNet_model.evaluate(ML.attributes_test, ML.targets_test_one_hot)
print(f'Test Accuracy: {test_accuracy:.4f}')
