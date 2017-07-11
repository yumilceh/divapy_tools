"""
Created on June,2017

@author: Juan Manuel Acevedo Valle
"""

from SensorimotorExploration.Algorithm.utils.functions import get_random_motor_set
from divapy import  Diva

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

import tensorflow as tf

use_device = '/cpu:0' # '/cpu:0' or '/gpu:0'
with tf.device(use_device):
    #this is the size of our encoded representations
    # encoding_dim = 8 # 8 floats -> compression of factos  max(220/8): 220 is the maximum lenght of af

    # input place holder
    encoding_dim =6
    input_af = Input(shape=(encoding_dim,))

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(220, activation='sigmoid')(input_af)

    # this model maps an input to its reconstruction
    decoder = Model(input_af, decoded)

    # create a place holder for an encoded (8-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the las later of the autoencoder model
    decoder_layer = decoder.layers[-1]

    decoder.compile(optimizer='RMSprop', loss='mean_squared_error')

    data = np.load('data/dataset1.npz')
    norm_af_train = data['norm_af_train']
    norm_af_test = data['norm_af_test']

    norm_ss_train = data['norm_ss_train']
    norm_ss_test = data['norm_ss_test']

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    decoder.fit(norm_ss_train, norm_af_train, epochs=15,
                    batch_size=250, shuffle=True,
                    validation_data=(norm_ss_test, norm_af_test))

    # encode and decode some digits
    # note that we take them from the *test* set
    decoded_afs = decoder.predict(norm_ss_test)

    n = 10  # how many digits we will display

    for i in range(n):
        plt.figure()#figsize=(20, 4))
        # display original
        ax = plt.subplot(2, 1, 1)
        plt.plot(norm_af_test[i,:])
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, 1, 2)
        plt.plot(decoded_afs[i,:])
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        plt.show(block=True)
