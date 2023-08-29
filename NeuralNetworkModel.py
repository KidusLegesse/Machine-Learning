import tensorflow as tf
import ML
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('categorical_crossentropy')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()


def train_model(X_train, y_train):
    model = tf.keras.Sequential()
    # input_dim is the number of features
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(13,)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    model.compile(optimizer="adam", loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        X_train, y_train, epochs=64, batch_size=32, validation_data=(ML.X_valid, ML.y_valid_one_hot), verbose=0
    )
    return model, history


model, history = train_model(ML.X_train, ML.y_train_one_hot)
plot_history(history)

val_loss, val_accuracy = model.evaluate(ML.X_valid, ML.y_valid_one_hot)
test_loss, test_accuracy = model.evaluate(ML.X_test, ML.y_test_one_hot)
print(f'Test Accuracy: {test_accuracy:.4f}')
