import tensorflow as tf
import numpy as np
import os
import cv2
from numpy import genfromtxt
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import h5py

    
def load_dataset():
    train_dataset = h5py.File('train_dataset.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_X"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_Y"][:]) # your train set labels
    m1, h1, w1, c1 = train_set_x_orig.shape

    test_dataset = h5py.File('test_dataset.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_X"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_Y"][:]) # your test set labels
    m3, h3, w3, c3 = test_set_x_orig.shape


    train_set_x_orig = np.reshape(train_set_x_orig, (m1,c1,h1,w1))
    
    test_set_x_orig = np.reshape(test_set_x_orig, (m3,c3,h3,w3))
    

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig 

def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):
    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_first', name=layer+'_conv'+num)(x)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding, data_format='channels_first')(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_first', name=layer+'_conv'+'2')(tensor)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor