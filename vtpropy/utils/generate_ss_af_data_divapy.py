"""
Created on June,2017

@author: Juan Manuel Acevedo Valle
"""

from exploration.algorithm.utils.functions import get_random_motor_set
from divapy import  Diva

import numpy as np

system = Diva()
system.n_motor = 13
system.min_motor_values = np.array([-3]*10+[0]*3)
system.max_motor_values = np.array([3]*10+[1]*3)

n_samples_train = 1000000
n_samples_test = 1000
motor_commands = get_random_motor_set(system, n_samples_train)
motor_commands_test = get_random_motor_set(system, n_samples_test)

af_train = np.zeros((n_samples_train, 220))
ss_train = np.zeros((n_samples_train, 6))
for i in range(n_samples_train):
    # print(motor_commands[i,:])
    a,ss,c,af = system.get_audsom(motor_commands[i,:])
    af_train[i,:len(af)] = af
    ss_train[i,:] = ss[:6]
    if i/n_samples_train in [0.2, 0.4, 0.6, 0.8]:
        print('{}% of training samples has been obtained'.format(i/n_samples_train*100))

af_test = np.zeros((n_samples_test, 220))
ss_test = np.zeros((n_samples_test, 6))
for i in range(n_samples_test):
    # print(motor_commands[i,:])
    a,ss,c,af = system.get_audsom(motor_commands_test[i,:])
    af_test[i,:len(af)] = af
    ss_test[i,:] = ss[:6]
    if i/n_samples_test in [0.2, 0.4, 0.6, 0.8]:
        print('{}% of test samples has been obtained'.format(i/n_samples_test*100))


af_max_val = np.maximum(np.max(af_test, axis=0), np.max(af_train, axis=0))
af_min_val = np.maximum(np.min(af_test, axis=0), np.min(af_train, axis=0))

norm_af_train = np.divide(np.subtract(af_train, af_min_val), np.subtract(af_max_val, af_min_val))
norm_af_train[np.where(np.isnan(norm_af_train))] = 0

norm_af_test = np.divide(np.subtract(af_test, af_min_val), np.subtract(af_max_val, af_min_val))
norm_af_test[np.where(np.isnan(norm_af_test))]=0

ss_max_val = np.maximum(np.max(ss_test, axis=0), np.max(ss_train, axis=0))
ss_min_val = np.maximum(np.min(ss_test, axis=0), np.min(ss_train, axis=0))

norm_ss_train = np.divide(np.subtract(ss_train, ss_min_val), np.subtract(ss_max_val, ss_min_val))
norm_ss_train[np.where(np.isnan(norm_ss_train))] = 0

norm_ss_test = np.divide(np.subtract(ss_test, ss_min_val), np.subtract(ss_max_val, ss_min_val))
norm_ss_test[np.where(np.isnan(norm_ss_test))]=0

np.savez("dataset1.npz", af_max_val=af_max_val,
                              af_min_val=af_min_val,
                              af_train=af_train,
                              norm_af_train=norm_af_train,
                              af_test=af_test,
                              norm_af_test=norm_af_test,
                              ss_max_val=ss_max_val,
                              ss_min_val=ss_min_val,
                              ss_train=ss_train,
                              norm_ss_train=norm_ss_train,
                              ss_test=ss_test,
                              norm_ss_test=norm_ss_test)