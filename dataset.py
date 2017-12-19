import numpy as np
from collections import Counter
import math

def load_data(datafile):
    """
    Load data into np.charArray
    :param datafile: path to file
    """
    return np.genfromtxt(datafile, dtype=np.str, delimiter=',')


def count_missing(data):
    """
    Count missing data per feature
    :param data: np.charArray with the data
    """
    res = []
    for i in range(len(data[0])):
        unique, counts = np.unique(data[:,i],return_counts=True)
        i, = np.where(unique == '?')
        if len(i) > 0:
            res.append(counts[i][0])
        else:
            res.append(0)
    return res

def remove_missing(data):
    """
    Remove datapoints if a feature value is missing
    :param data: np.charArray with the data
    """
    res = []
    for i in range(len(data)):
        j, = np.where(data[i] == '?')
        if len(j) == 0:
            res.append(data[i].tolist())
    return np.array(res)

def unique_vals(data):
    """
    Count unique values per feature
    :param data: np.charArray with the data
    """
    res = []
    for i in range(len(data[0])):
        unique, counts = np.unique(data[:,i],return_counts=True)
        res.append(len(unique))
    return res


def count_occurence(data_column):
    """
    Count occurence of all unique values for one feature
    :param data: colomn of np.charArray with the data (feature)
    """
    res = {}
    unique, counts = np.unique(data_column, return_counts=True)
    for i in range(len(unique)):
        res[unique[i]] = counts[i]
    return res



def validation_split(data, ratio):
    """
    Randomly split data into training and validation set with ratio
    :param data: np.charArray with the data
    :param ratio: ratio for the split
    """
    np.random.seed(42)
    np.random.shuffle(data)
    tmp = int(math.floor(len(data)*ratio))
    return data[0:tmp], data[tmp:len(data)]


def x_y_split(data):
    """
    Split data into X and Y
    :param data: np.charArray with the data
    """
    return data[:,0:len(data[0])-1], data[:,len(data[0])-1]


def disc_num_split(data, thres=20):
    """
    Split data columns into discrete and numeric
    :param data: np.charArray with the data
    :param thres: treshold for number of unique values
    """
    data_int_i = []
    data_float_i = []
    u = unique_vals(data)
    for i in range(len(u)):
        if u[i] < thres: # integer
            data_int_i.append(i)
        else: # float
            data_float_i.append(i)

    data_int = []
    for i in data_int_i:
        data_int.append(data[:,i].tolist())
    data_float = []
    for i in data_float_i:
        data_float.append(data[:,i].tolist())
    return np.array(data_int).T, np.array(data_float).T


def get_clean_data():
    """
    Load in the data, get a 2d np array with on each row a patient.
    """
    data_c = load_data("processed.cleveland.data")
    data_v = load_data("processed.va.data")
    data_h = load_data("processed.hungarian.data")
    data_s = load_data("processed.switzerland.data")
    data = np.concatenate((data_c, data_h, data_s, data_v), axis=0)
    data = remove_missing(data)
    return data.astype(np.float64)
