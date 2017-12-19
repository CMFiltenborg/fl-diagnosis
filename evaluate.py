import numpy as np
from dataset import *
from rule_generator import *
from membership import *
import pickle
from sklearn.utils import shuffle
import itertools


def validate_sys(reasoner, test_data, thresh):
    """
    Simpler version of validate_sys_cv, does not deploy CV to rigorously test the system.
    """
    test_x, test_y = x_y_split(test_data)
    correct = 0
    result = np.zeros((len(test_x), 2))
    confusion_matrix = np.zeros((5, 5))
    non_classified = 0

    for i, x in enumerate(test_x):
        prediction = reasoner.inference(x)
        y = int(test_y[i])
        result[i] = [y, prediction]
        if prediction is None:
            non_classified += 1
            continue

        prediction_rounded = int(round(prediction))
        confusion_matrix[prediction_rounded, y] += 1

        if abs(y - prediction) < thresh:
            correct += 1

    percentage_correct = correct / len(test_data) * 100

    print(result, result.shape)
    print(confusion_matrix)
    print('{} total test examples'.format(len(test_data)))
    print('Had {} test examples that could not be classified by the rules'.format(non_classified))
    print('Percentage correct: {}%'.format(percentage_correct))

    return percentage_correct


def validate_sys_cv(df, mf_ns):
    """
    Evaluates the FLS using CV.
    """
    test_n = 0
    k_folds = 5
    print('Total tests: {}'.format(len(columns) * k_folds))
    k_fold_scores = np.zeros((k_folds))
    confusion_matrix_total = np.zeros((5, 5))
    non_classified_total = 0
    k = 0
    print([i for i in df])
    for train_df, test_df in cv_splits(df, k_folds=k_folds):
        test_x, test_y = x_y_split(test_df.as_matrix())
        confusion_matrix = np.zeros((5, 5))

        # get input and output variables and their membership functions
        # Test a different n for every input variable
        # Temporary ns variable for this test
        inputs, output = get_variables(train_df, mf_ns)
        rulebase = generate_rules(train_df, inputs, output, 'AND')
        reasoner = Reasoner(rulebase, inputs, output, 1000)

        correct = 0
        non_classified = 0
        if test_n % 10 == 0:
            print('Test {}'.format(test_n))

        for i, x in enumerate(test_x):
            prediction = reasoner.inference(x)
            if prediction is None:
                non_classified += 1
                non_classified_total += 1
                continue

            y = int(test_y[i])
            prediction_rounded = int(round(prediction))
            confusion_matrix[prediction_rounded, y] += 1
            confusion_matrix_total[prediction_rounded, y] += 1

            if abs(y - prediction) < thresh:
                correct += 1

        test_n += 1
        percentage_correct = correct / len(test_df) * 100
        print(percentage_correct)
        if percentage_correct == 0:
            break

        k_fold_scores[k] = percentage_correct

        k += 1

        # print(result, result.shape)
        print(confusion_matrix)
        print('{} total examples, had {} test examples that could not be classified by the rules'.format(len(test_x),
                                                                                                         non_classified))
        print('Percentage correct: {}%'.format(percentage_correct))

    print('Total confusion matrix: \n', confusion_matrix_total)
    print('K fold scores: \n', k_fold_scores)
    print('Total non classified: {}'.format(non_classified_total))


def grid_search_antecedents(test_data, thresh):
    """
    Tries removing clauses from the fuzzy rules, to create more general rules
    which theoretically could lead to a better performance of the FLS.
    However this is hard to implement with CV as the rule set is different everytime,
    with the sparse amount of data for this project this does not work.
    """
    df = pd.DataFrame(train, columns=columns)
    inputs, output = get_variables(df)  # get input and output variables and their membership functions

    test_x, test_y = x_y_split(test_data)

    rulebase = generate_rules(df, inputs, output)
    unpacked_rules = rulebase.rules
    reasoner = Reasoner(rulebase, inputs, output, 200)
    best_score = 0
    test_n = 0
    max_test_n = 5000
    removed_antecedents = []
    print('Total antecedents {}'.format(sum([len(rule.antecedent) for rule in unpacked_rules])))

    stop = False
    for i in range(len(unpacked_rules)):
        if stop:
            break

        rule = unpacked_rules[i]
        for j in range(len(rule.antecedent)):
            test_antecedent = rule.antecedent[j]
            rule.antecedent[j] = ''

            correct = 0
            result = np.zeros((len(test_x), 2))
            confusion_matrix = np.zeros((5, 5))
            non_classified = 0
            if test_n % 10 == 0:
                print('Test {}'.format(test_n))

            for i, x in enumerate(test_x):
                prediction = reasoner.inference(x)
                if prediction is None:
                    non_classified += 1
                    continue

                y = int(test_y[i])
                result[i] = [y, prediction]
                prediction_rounded = int(round(prediction))
                confusion_matrix[prediction_rounded, y] += 1

                if abs(y - prediction) < thresh:
                    correct += 1

            test_n += 1
            percentage_correct = correct / len(test_data) * 100
            if percentage_correct > best_score:
                print('Found a better score: {} > {}'.format(percentage_correct, best_score))
                print('Removed {}'.format(test_antecedent))

                best_score = percentage_correct
                removed_antecedents.append(test_antecedent)
            if percentage_correct < best_score:
                rule.antecedent[j] = test_antecedent

            if test_n >= max_test_n:
                stop = True
                break

    new_rule_base = Rulebase(unpacked_rules)
    with open('rulebase.pkl', 'wb') as output:
        pickle.dump(new_rule_base, output, pickle.HIGHEST_PROTOCOL)

    return new_rule_base


def grid_search_n_mf(df):
    """
    Optimizes the system by finding the optimal n membership functions for every input variable.
    Uses cross validation.
    """
    best_score = 0
    test_n = 0
    n_mf_tests = [3, 5, 7, 9, 11, 13, 15]
    k_folds = 5
    saved_ns = {}
    print('Total tests: {}'.format(df.shape[1] * len(n_mf_tests) * k_folds))

    for column in df:
        for i in range(len(n_mf_tests)):
            n_test = n_mf_tests[i]
            k_fold_scores = np.zeros((k_folds))

            k = 0
            for train_df, test_df in cv_splits(df, k_folds=k_folds):
                test_x, test_y = x_y_split(test_df.as_matrix())

                # get input and output variables and their membership functions
                # Test a different n for every input variable
                # Temporary ns variable for this test
                ns = {**saved_ns}
                ns[column] = n_test
                inputs, output = get_variables(train_df, ns)
                rulebase = generate_rules(train_df, inputs, output, 'AND')
                reasoner = Reasoner(rulebase, inputs, output, 1000)

                correct = 0
                non_classified = 0
                if test_n % 10 == 0:
                    print('Test {}'.format(test_n))

                for i, x in enumerate(test_x):
                    prediction = reasoner.inference(x)
                    if prediction is None:
                        non_classified += 1
                        continue

                    y = int(test_y[i])
                    if abs(y - prediction) < thresh:
                        correct += 1

                test_n += 1
                percentage_correct = correct / len(test_df) * 100
                if percentage_correct < best_score:
                    break

                k_fold_scores[k] = percentage_correct
                k += 1

            min = np.min(k_fold_scores)
            if min > best_score:
                print('Found a better score: {} > {}'.format(min, best_score))
                old_n = saved_ns[column] if column in saved_ns else 5
                print('New n:{}, old n: {}, variable: {}'.format(n_test, old_n, column))
                saved_ns[column] = n_test
                best_score = min

    # Previously used to save the rulebase to disk
    # with open('rulebase.pkl', 'wb') as output:
    #     pickle.dump(new_rule_base, output, pickle.HIGHEST_PROTOCOL)

    print('Best N amount of membership functions: {}'.format(saved_ns))
    return inputs, output


def cv_splits(df, k_folds):
    """
    Splits the data k times and returns a iterator of these sets
    Implementation splitting a dataset for k-fold cross validation
    :param df: pandas Dataframe of the data to use
    :param k_folds: The amount of splits to do on the data
    """
    df = shuffle(df)
    test_start = 0
    test_end = 0
    for k in range(k_folds):
        if k != 0:
            test_start += 1 / k_folds

        if k != k_folds:
            test_end += 1 / k_folds

        rows = df.shape[0]
        from_test = math.ceil(rows * test_start)
        till_test = math.ceil(rows * test_end)

        test_indices = df.index[from_test:till_test]

        test_data = df[from_test:till_test]
        data = df.drop(test_indices)
        train_data = data

        yield train_data, test_data


def grid_search_inputs(df):
    """
    Uses cross-validation to search for the optimal combination of input features
    """
    input_tests = []
    # Create every combination of features possible (starting from atleast 3 features)
    for i in range(3, len(columns_without_y)):
        combis = itertools.combinations(columns_without_y, i)
        for combination in combis:
            combination = list(combination)
            if y_column not in combination:
                combination.append(y_column)
            input_tests.append(combination)

    best_score = 0
    test_n = 0
    k_folds = 5
    print('Grid search inputs: Total tests: {}'.format(len(input_tests) * k_folds))

    for i in range(len(input_tests)):
        input_columns = input_tests[i]

        k_fold_scores = np.zeros((k_folds))

        k = 0
        for train_df, test_df in cv_splits(df, k_folds=k_folds):
            train_df = train_df[input_columns]
            test_x, test_y = x_y_split(test_df.as_matrix())

            # get input and output variables and their membership functions
            # Test a different n for every input variable
            # Temporary ns variable for this test
            inputs, output = get_variables(train_df)
            rulebase = generate_rules(train_df, inputs, output, 'AND')
            reasoner = Reasoner(rulebase, inputs, output, 1000)

            correct = 0
            non_classified = 0
            if test_n % 10 == 0:
                print('Test {}'.format(test_n))

            for i, x in enumerate(test_x):
                prediction = reasoner.inference(x)
                if prediction is None:
                    non_classified += 1
                    continue

                y = int(test_y[i])
                if abs(y - prediction) < thresh:
                    correct += 1

            test_n += 1
            percentage_correct = correct / len(test_df) * 100
            if percentage_correct < best_score:
                break

            k_fold_scores[k] = percentage_correct
            k += 1

        min = np.min(k_fold_scores)
        if min > best_score:
            print('Found a better score: {} > {}'.format(min, best_score))
            best_score = min
            best_columns = input_columns
            print(best_columns)

    print('Best features: {}'.format(best_columns))
    return best_columns


if __name__ == '__main__':
    np.random.seed(42)
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
    columns_without_y = [
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
        # '14. #58 (num)',
    ]
    ratio = 0.7
    # train, test = validation_split(data, ratio)
    thresh = 0.5

    df = pd.DataFrame(data, columns=columns)

    grid_search_inputs(df)
    # inputs, output = get_variables(df, mf_ns)  # get input and output variables and their memebership functions
    # rulebase = grid_search_antecedents(test, 0.5)
    # plot_mfs([output])
    # rulebase = generate_rules(df, inputs, output)  # generate rules

    df = df[['1. #3 (age)', '2. #4 (sex)', '3. #9 (cp)', '14. #58 (num)']]
    # inputs, output = grid_search_n_mf(df)
    # mf_ns = {'1. #3 (age)': 7}
    mf_ns = {'1. #3 (age)': 15, '3. #9 (cp)': 15, '14. #58 (num)': 11}
    # mf_ns = {}
    validate_sys_cv(df, mf_ns)

    # with open('rulebase.pkl', 'rb') as input:
    # rulebase = pickle.load(input)

    # thinker = Reasoner(rulebase, inputs, output, 200) # make a Reasoner object to initialize the whole system.
    # validate_sys(thinker, test, 0.5)
