import numpy as np
import pandas as pd
import math
from dataset import *
from fuzzy_functions import *
from rule_generator import *


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


def validate_sys(reasoner, test_data, thresh):
    test_x, test_y = x_y_split(test_data)
    correct = 0
    result = np.zeros((len(test_x), 2))
    for i, x in enumerate(test_x):
        r = reasoner.inference(x)
        y = test_y[i]
        result[i] = [y, r]
        if r and abs(y - r) < thresh:
            correct += 1

    print(result, result.shape)
    return correct / len(test_data) * 100


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

    ratio = 0.7
    train, test = validation_split(data, ratio)

    df = pd.DataFrame(train, columns=columns)
    # print(df.head())
    inputs, output = get_variables(df)  # get input and output variables and their memebership functions
    rulebase = generate_rules(df, inputs, output)  # generate rules
    thinker = Reasoner(rulebase, inputs, output, 200)  # make a Reasoner object to initialize the whole system.
    # datapoint = [100, 1]
    # print(round(thinker.inference(datapoint)))
    # print(df.shape)
    print(validate_sys(thinker, test, 0.5))


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
