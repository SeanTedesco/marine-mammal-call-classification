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

import os
import librosa
import math
import json

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


# test size = 0.8, validation size = 0.3
def prepare_datasets(X, y, test_size, validation_size):    
    # create the train/test split
    # so this means 0.8 of original data size (5*100 samples) is for test, 0.2 is for train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # create the train/validation split
    # so 0.7*0.2 is for train, 0.3*0.2 is for validation 
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
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    
    # Save image
    fig.subplots_adjust(hspace=0.5)
    plt.savefig(image_name, bbox_inches='tight')
    plt.show()

def predict(model, X, y):
    X = X[np.newaxis, ...] # to put make a 4D
    
    # prediction = [ [0.1, 0.2, ...] ] result of the softmax
    prediction = model.predict(X) # X -> (130, 13, 1) but expect 4D i.e num samples (1, 130, 13, 1)
    
    # extract the index with the max value
    predicted_index = np.argmax(prediction, axis=1) # [idx] 
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))
#     print(X.shape)

def count_trainable_parameters(mdl):
    from keras.utils.layer_utils import count_params
    trainable_count = count_params(mdl.trainable_weights)
    non_trainable_count = count_params(mdl.non_trainable_weights)
    return trainable_count


def add_conv_layer(mdl, input_shape, output_shape):
    mdl.add(keras.layers.Conv2D(output_shape, (3, 3), activation='relu', input_shape=input_shape))
    mdl.add(keras.layers.MaxPool2D((3, 3), strides=(1,1), padding='valid'))
    mdl.add(keras.layers.BatchNormalization()) # normalizes the activation at the layer, speeds up training


def add_dense_flat_layer(mdl):
    # flatten output and feed it into dense layer
    mdl.add(keras.layers.Flatten()) # flatten conv output
    mdl.add(keras.layers.Dense(64, activation='relu'))
    mdl.add(keras.layers.Dropout(0.1)) # randomly drops neurons


def add_softmax_layer(mdl):
    # output layer that uses softmax
    mdl.add(keras.layers.Dense(9, activation='softmax')) # number of neurons of the classifications we want to predict


def create_layered_cnn(n_layers, input_shape, output_shape):
    model = keras.Sequential()
    for n in range(n_layers):
        add_conv_layer(model, input_shape, output_shape)
    add_dense_flat_layer(model)
    add_softmax_layer(model)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

    model.build(input_shape)

    return model


#### Gradio stuff ####
SAMPLE_RATE = 22050 # how many samples per unit to get discrete signal, so 22050 samples per second.
DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION # 661500 samples per track

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }

    # each track is divided into 10 segments
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # 1.2 -> 2
    
    # loop through all the animals
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure that we're not at the root level
        if dirpath is not dataset_path:
            
            # save the semantic label
            dirpath_components = dirpath.split("/") # animals/killerwhale => ['animals', 'killerwhale', 'humpback']
            semantic_label = dirpath_components[-1]
            if semantic_label not in ('.ipynb_checkpoints'):
                data['mapping'].append(semantic_label)
                print("\nProcessing {}".format(semantic_label))
            
            
            # process files for a specific marine animal
            for f in filenames:
                
                # load the audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # process segments extracting mfcc and storing the data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s =0 -> 0
                    finish_sample = start_sample + num_samples_per_segment # s=0 -> num_samples_per_segment
                    
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                               sr=sr, 
                                               n_fft=n_fft,
                                               n_mfcc=n_mfcc,
                                               hop_length=hop_length)
                    
                    mfcc = mfcc.T
                    
                    # store mfcc for segment if it has the expected length 
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment: #  and semantic_label not in ('.ipynb_checkpoints')
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(i)
                        print("{}, segment:{}".format(file_path, s+1))
                    
                
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)