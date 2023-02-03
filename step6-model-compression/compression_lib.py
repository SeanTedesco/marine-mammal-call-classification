'''
Model Compression Library
===========================

This library provides helper functions to compress,
quantize, and calculate model sizes.
'''

################################################################################
# IMPORTS
#
import tempfile
import json
import numpy as np
from sklearn.model_selection import train_test_split

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

def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)