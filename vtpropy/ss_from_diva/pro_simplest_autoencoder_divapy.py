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
    encoding_dim = 6 # 8 floats -> compression of factos  max(220/8): 220 is the maximum lenght of af

    # input place holder
    input_af = Input(shape=(220,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_af)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(220, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_af, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_af, encoded)

    # create a place holder for an encoded (8-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the las later of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


    autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')


    data = np.load('data/dataset1.npz')
    norm_af_train = data['norm_af_train']
    norm_af_test = data['norm_af_test']


    autoencoder.fit(norm_af_train, norm_af_train, epochs=15,
                    batch_size=250, shuffle=True,
                    validation_data=(norm_af_test, norm_af_test))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_afs = encoder.predict(norm_af_test)
    decoded_afs = decoder.predict(encoded_afs)

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
