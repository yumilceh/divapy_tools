"""
Created on June,2017

@author: Juan Manuel Acevedo Valle
"""
import numpy as np

def abs_sqrt(af, tresh):
    af_ = af + 0.0
    ii=-1
    while af_[ii] == 0:
        af_[ii] = tresh + 0.01
        ii -= 1
    af__ = np.absolute(af_ - tresh)
    af_ = ((af_ - tresh)- af__)/2.
    return np.square(af_)
