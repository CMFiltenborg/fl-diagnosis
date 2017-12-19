import numpy as np
import pandas as pd
import math
from dataset import *
from fuzzy_functions import *
import matplotlib.pyplot as plt

y_column = '14. #58 (num)'

def get_variables(df, mf_ns={}):
    """
    :param df: DataFrame containing all features data
    :param mf_ns: A dictionary that contains how many membership functions should be created
    for each fuzzy variable. If a key is not set the default of 5 is used.
    :return: (Input[], Output): tuple of input and output variables
    """
    y_n = 5
    if y_column in mf_ns:
        y_n = mf_ns[y_column]

    y = df[y_column].as_matrix()
    df = df.drop(y_column, 1)

    output_var = Output('output', (y.min(), y.max()), make_membership_functions(y_column, y, y_n))
    input_vars = []
    for column_name in df:
        n = 5
        if column_name in mf_ns:
            n = mf_ns[column_name]

        feature = df[column_name].as_matrix()
        membership_functions = make_membership_functions(column_name, feature, n)
        x_variable = Input(column_name, (feature.min(), feature.max()), membership_functions)

        input_vars.append(x_variable)

    return input_vars, output_var


def make_membership_functions(feature_name, feature_vec, n):
    """
    :param feature_name: Name of the particular feature
    :param feature_vec: vector of values for one feature
    :param n: number of memberships for the feature
    :return TriangularMF[]: array of membership functions
    """
    min_range = feature_vec.min()
    max_range = feature_vec.max()

    # Initialize parameters
    mid_mf = int(math.ceil(n / 2))
    increment = (max_range - min_range) / (n - 1)
    start = min_range
    top = min_range
    end = min_range + increment

    membership_functions = []
    # Create n membership functions
    for i in range(n):
        name = determine_mf_name(feature_name, i, mid_mf)
        mf = TriangularMF(name, start, top, end)
        membership_functions.append(mf)

        # Update MF positioning
        if i != 0:
            start += increment
        if top != max_range:
            top += increment
        if end != max_range:
            end += increment

    return membership_functions


def determine_mf_name(feature_name, i, mid_mf):
    """
    :param feature_name: Name of the particular feature
    :param i: feature number
    :param mid_mf: middle of MF
    :return name: name of MF
    """
    name = ''
    if i + 1 < mid_mf:
        name = '{}: Small {}'.format(feature_name, i + 1)
    if i + 1 == mid_mf:
        name = '{}: CE'.format(feature_name)
    if i + 1 > mid_mf:
        name = '{}: Big {}'.format(feature_name, i + 1)

    return name


def plot_mfs(variable_list):
    """
    plot MFs for variables
    :param variable_list: list of variable objects
    """
    color = {0:"bo",1:"ro",2:"go",3:"co", 4:"yo", 5:"mo"}
    l = len(variable_list)
    _, ax = plt.subplots(l)
    for i, var in enumerate(variable_list):
        t1, t2 = var.range
        x = np.linspace(t1,t2,500)
        for j, mf in enumerate(var.mfs):
            y = []
            for xx in x:
                y.append(mf.calculate_membership(xx))
            ax[i].plot(x,y,color[j])
    plt.show()
