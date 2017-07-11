"""
Created on June,2017

@author: Juan Manuel Acevedo Valle
"""

from SensorimotorExploration.Algorithm.utils.functions import get_random_motor_set
from divapy import  Diva

import numpy as np
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

import tensorflow as tf

use_device = '/cpu:0' # '/cpu:0' or '/gpu:0'
with tf.device(use_device):
    #this is the size of our encoded representations
    encoding_dim = 6 # 8 floats -> compression of factos  max(220/8): 220 is the maximum lenght of af
    layer1_dim = 70
    layer2_dim =35


    # input place holder
    input_af = Input(shape=(220,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(layer1_dim, activation = 'relu')(input_af)
    encoded = Dense(layer2_dim, activation = 'relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(layer2_dim, activation='sigmoid')(encoded)
    decoded = Dense(layer1_dim, activation='sigmoid')(decoded)
    decoded = Dense(220, activation='sigmoid')(decoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_af, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_af, encoded)

    # create a place holder for an encoded (8-dimensional) input
    encoded_input1 = Input(shape=(encoding_dim,))
    decoder_layer1 = autoencoder.layers[-3]
    decoder1 = Model(encoded_input1, decoder_layer1(encoded_input1))
    encoded_input2 = Input(shape=(layer2_dim,))
    decoder_layer2 = autoencoder.layers[-2]
    decoder2 = Model(encoded_input2, decoder_layer2(encoded_input2))
    encoded_input3 = Input(shape=(layer1_dim,))
    decoder_layer3 = autoencoder.layers[-1]
    decoder3 = Model(encoded_input3, decoder_layer3(encoded_input3))

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')


    data = np.load('data/dataset1_contact.npz')
    norm_contact_train = data['norm_contact_train']
    norm_contact_test = data['norm_contact_test']

    autoencoder.fit(norm_contact_train, norm_contact_train, epochs=15,
                    batch_size=250, shuffle=True,
                    validation_data=(norm_contact_test, norm_contact_test))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_contacts = encoder.predict(norm_contact_test)
    decoded_contacts1 = decoder1.predict(encoded_contacts)
    decoded_contacts2 = decoder2.predict(decoded_contacts1)
    decoded_contacts = decoder3.predict(decoded_contacts2)

    n = 10  # how many digits we will display

    for i in range(n):
        plt.figure()#figsize=(20, 4))
        # display original
        ax = plt.subplot(2, 1, 1)
        plt.plot(norm_contact_test[i,:])
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, 1, 2)
        plt.plot(decoded_contacts[i,:])
        plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        plt.show(block=True)
