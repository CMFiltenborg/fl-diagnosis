import numpy as np
from collections import Counter


def load_data(datafile):
    return np.genfromtxt(datafile, dtype=np.str, delimiter=',')


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



def validation_split(data, ratio):
    return data[0:math.floor(len(data)*ratio)], data[math.floor(len(data)*ratio):len(data)]


def x_y_split(data):
    res = []
    for i in range(len(data)):
        if data[i][len(data[0])-1] == '0':
            res.append(False)
        else:
            res.append(True)
    return data[:,0:len(data[0])-1], np.array(res)


def disc_num_split(data, thres=20):
    data_int_i = []
    data_float_i = []
    u = unique_vals(data)
    for i in range(len(u)):
        if u[i] < thres: # int
            data_int_i.append(i)
        else: #float
            data_float_i.append(i)
    data_int = []
    for i in data_int_i:
        data_int.append(data[:,i].tolist())
    data_float = []
    for i in data_float_i:
        data_float.append(data[:,i].tolist())
    return np.array(data_int).T, np.array(data_float).T


''' get a 2d np array with on each row a patient'''
def get_clean_data():
    data_c = load_data("processed.cleveland.data")
    data_v = load_data("processed.va.data")
    data_h = load_data("processed.hungarian.data")
    data_s = load_data("processed.switzerland.data")
    data = np.concatenate((data_c, data_h, data_s, data_v), axis=0)
    data = remove_missing(data)
    return data.astype(np.float64)
