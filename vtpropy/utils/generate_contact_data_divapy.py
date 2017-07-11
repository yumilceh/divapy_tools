"""
Created on June,2017

@author: Juan Manuel Acevedo Valle
"""
import numpy as np

from contact_functions import  abs_sqrt as contact_fuction

data = np.load('data/dataset1.npz')
af_train = data['af_train']
af_test = data['af_test']

tresh = 0.2

contact_train = np.zeros(af_train.shape)
contact_test = np.zeros(af_test.shape)
for i in range(af_train.shape[0]):
    contact_train[i,:] = contact_fuction(af_train[i,:], tresh)
for i in range(af_test.shape[0]):
    contact_test[i,:] = contact_fuction(af_test[i,:], tresh)

contact_max_val = np.maximum(np.max(contact_test, axis=0), np.max(contact_train, axis=0)) + 0.1
contact_min_val = np.maximum(np.min(contact_test, axis=0), np.min(contact_train, axis=0)) - 0.1

norm_contact_train = np.divide(np.subtract(contact_train, contact_min_val), np.subtract(contact_max_val, contact_min_val))
norm_contact_train[np.where(np.isnan(norm_contact_train))] = 0

norm_contact_test = np.divide(np.subtract(contact_test, contact_min_val), np.subtract(contact_max_val, contact_min_val))
norm_contact_test[np.where(np.isnan(norm_contact_test))]=0


np.savez("data/dataset1_contact.npz", contact_max_val=contact_max_val,
                              contact_min_val=contact_min_val,
                              contact_train=contact_train,
                              norm_contact_train=norm_contact_train,
                              contact_test=contact_test,
                              norm_contact_test=norm_contact_test)