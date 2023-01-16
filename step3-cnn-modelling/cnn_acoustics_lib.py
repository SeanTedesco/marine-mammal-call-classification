'''
CNN Acoustics Library
===========================

This library provides helper functions to run CNN 
models on acoustic signals represented by MFCCs.
'''

################################################################################
# IMPORTS
#
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

################################################################################
# FUNCTIONS
#
def load_cnn_json(mfccs_json_path:str):
    with open(mfccs_json_path, "r") as fp:
        data = json.load(fp)

        # convert list into numpy arrays
        y = np.array(data['labels'])
        X = np.array(data['mfcc'])
        L = np.array(data['mapping'])

    return (X, y, L)


def prepare_datasets(X, y, test_size, validation_size):    
    # create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # create the train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    # Tensorflow for CNN expect a 3D array -> (130, 13, 1) audio grayscale images
    X_train = X_train[..., np.newaxis] # 4d array -> (num of samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis] 
    X_test = X_test[..., np.newaxis] 
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def plot_history(history, image_name:str):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    
    # Save image
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(image_name, bbox_inches='tight')
    plt.show()


def build_model(input_shape):
    # Instantiate model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization()) # normalizes the activation at the layer, speeds up training

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten()) # flatten conv output
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3)) # randomly drops neurons
            
    # output layer that uses softmax
    model.add(keras.layers.Dense(9, activation='softmax')) # number of neurons of the classifications we want to predict

    return model


def predict(model, X, y):
    X = X[np.newaxis, ...] # to put make a 4D
    
    # prediction = [ [0.1, 0.2, ...] ] result of the softmax
    prediction = model.predict(X) # X -> (130, 13, 1) but expect 4D i.e num samples (1, 130, 13, 1)
    
    # extract the index with the max value
    predicted_index = np.argmax(prediction, axis=1) # [idx] 
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))
#     print(X.shape)