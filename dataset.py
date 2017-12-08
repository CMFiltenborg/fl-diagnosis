import numpy as np
from collections import Counter


def load_data(datafile):
    return np.loadtxt(datafile, dtype=np.str, delimiter=',')


def count_missing(data):
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
    res = []
    for i in range(len(data)):
        j, = np.where(data[i] == '?')
        if len(j) == 0:
            res.append(data[i].tolist())
    return np.array(res)

def unique_vals(data):
    res = []
    for i in range(len(data[0])):
        unique, counts = np.unique(data[:,i],return_counts=True)
        res.append(len(unique))
    return res


def count_occurence(data_column):
    res = {}
    unique, counts = np.unique(data_column, return_counts=True)
    for i in range(len(unique)):
        res[unique[i]] = counts[i]
    return res

''' get a 2d np array with on each row a patient'''
def get_clean_data():
    data_c = load_data("processed.cleveland.data")
    data_v = load_data("processed.va.data")
    data_h = load_data("processed.hungarian.data")
    data_s = load_data("processed.switzerland.data")
    data = np.concatenate((data_c, data_h, data_s, data_v), axis=0)
    return remove_missing(data)
