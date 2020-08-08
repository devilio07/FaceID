from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils import *
from inception_model import *

#FaceID = faceRecoModel(input_shape=(3, 200, 200))

train_X, train_Y, test_X, test_Y = load_dataset()

def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    pos_dist = tf.reduce_sum((anchor - positive)**2, axis=-1)
    neg_dist = tf.reduce_sum((anchor - negative)**2, axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    return loss

print(train_Y.shape)
#FaceID.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

#FaceID.fit(train_X, train_Y, batch_size = 16, epochs = 3)