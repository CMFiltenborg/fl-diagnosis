import numpy as np
import pandas as pd
import math
from dataset import *
from fuzzy_functions import *


def get_variables(df):
    """
    :param df: DataFrame containing all features data
    :return: (Input[], Output): tuple of input and output variables
    """
    y_column = '14. #58 (num)'
    y = df[y_column].as_matrix()
    df = df.drop(y_column, 1)

    output_var = Output('output', (y.min(), y.max()), make_membership_functions(y_column, y, 5))
    input_vars = []
    for column_name in df:
        feature = df[column_name].as_matrix()
        membership_functions = make_membership_functions(column_name, feature, 5)
        x_variable = Input(column_name, (feature.min(), feature.max()), membership_functions)

        input_vars.append(x_variable)

    return input_vars, output_var


def make_membership_functions(feature_name, feature_vec, n):
    """
    :param feature_vec: vector of values for one feature
    :param n: number of memberships for the feature
    :return TriangularMF[]: array of membership functions
    """
    # r = [min(feature_vec), max(feature_vec)]
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
    name = ''
    if i + 1 < mid_mf:
        name = '{}: Small {}'.format(feature_name, i + 1)
    if i + 1 == mid_mf:
        name = '{}: CE'.format(feature_name)
    if i + 1 > mid_mf:
        name = '{}: Big {}'.format(feature_name, i + 1)

    return name
