from tempfile import gettempdir
from subprocess import call
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from scipy.signal import butter, lfilter, lfilter_zi

def update_buffer(data_buffer, new_data):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return np.array(new_buffer)


def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """
    new_buffer = data_buffer[(int(data_buffer.shape[0] - newest_samples)):, :]                 

    return new_buffer

def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n
