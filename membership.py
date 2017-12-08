import numpy as np
import pandas as pd
from dataset import get_clean_data
from fuzzy_functions import *


def all_mfs(data, n=5):
    if data.ndim == 1:
        return make_membership(data, n)

    return [make_membership(data[:, i], n) for i in range(data.shape[1])]


def get_variables(df):
    """
    :param df: DataFrame containing all features data
    :return: (Input[], Output): tuple of input and output variables
    """
    y_column = '14. #58 (num)'
    y = df[y_column].as_matrix()
    df = df.drop(y_column, 1)

    output = Output('output', y.max() - y.min(), all_mfs(y, 5))
    input_vars = []
    for column_name in df:
        feature = df[column_name].as_matrix()
        x_variable = Input(column_name, feature.max() - feature.min(), all_mfs(feature, 5))

        input_vars.append(x_variable)

    return input_vars, output


def make_membership(feature_vec, n):
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
        name = determine_mf_name(i, mid_mf)
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


def determine_mf_name(i, mid_mf):
    name = ''
    if i + 1 < mid_mf:
        name = 'Small {}'.format(i + 1)
    if i + 1 == mid_mf:
        name = 'CE'
    if i + 1 > mid_mf:
        name = 'Big {}'.format(i + 1)

    return name


if __name__ == '__main__':
    data = get_clean_data()
    columns = [
        '1. #3 (age)',
        '2. #4 (sex)',
        '3. #9 (cp)',
        '4. #10 (trestbps)',
        '5. #12 (chol)',
        '6. #16 (fbs)',
        '7. #19 (restecg)',
        '8. #32 (thalach)',
        '9. #38 (exang)',
        '10. #40 (oldpeak)',
        '11. #41 (slope)',
        '12. #44 (ca)',
        '13. #51 (thal)',
        '14. #58 (num)',
    ]

    data = pd.DataFrame(data, columns=columns)
    print(data.head())
    get_variables(data)
    print(data.shape)

    # column = data[:,0]
    # mfs = make_membership(column, 3)

    # mfs = all_mfs(data)
    # print(mfs)
    # print(len(mfs))

    # data = np.array([[0, 10]])
    # data = data.T
    # print(data.shape)
    # mfs = all_mfs(data)
    #
    # print(mfs)
    # print(len(mfs[0]))
    # mfs = mfs[0]
    # for mf in mfs:
    #     print('name:[{}], start:{}, top:{}, end:{}'.format(mf.name, mf.start, mf.top, mf.end))
