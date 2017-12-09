import functools
import math
import numpy as np
from collections import defaultdict, Counter
from dataset import get_clean_data
from fuzzy_functions import *
import pandas as pd
from membership import get_variables

dataset = get_clean_data()
print(dataset)


def generate_rules(df, input_variables, output_variable):
    """
        Generates a rule for each input and output pair
        :param df:
        :type df: pd.DataFrame
        :param input_variables: All input variables
        :param output_variable: Single output variable
        :return: RuleBase
    """

    data = df.as_matrix()
    columns_cnt = data.shape[1]

    # dict where the key is the antecedent and
    # the value is a tuple of the (rule, degree)
    rules = {}
    rule_n = 1

    for i in range(data.shape[0]):
        antecedent = {}
        consequent = {}

        for j in range(columns_cnt):
            x = data[i, j]

            # Determine if it is a input or output var
            # Take the correct FL variable object with this data point
            if j < (columns_cnt - 1):
                var = input_variables[j]  # type: Input
            else:
                var = output_variable  # type: Output
            member_ship_degrees = var.calculate_memberships(x)
            mf_name = max(member_ship_degrees.keys(), key=(lambda k: member_ship_degrees[k]))

            if type(var) is Input:
                antecedent[mf_name] = member_ship_degrees[mf_name]
            else:
                consequent[mf_name] = member_ship_degrees[mf_name]

        # Create a key for the antecedent so we do not
        # have to do a lookup of ALL the rules created so far if there is a rule 'conflict'
        antecedent_key = '-'.join(antecedent)
        # Degree is the multiplied degree of all mf's
        degree = functools.reduce(multiply, list(antecedent.values()) + list(consequent.values()), 1)
        new_rule = Rule(rule_n, list(antecedent.keys()), "and", list(consequent.keys()))

        if antecedent_key in rules:
            current_rule, current_degree = rules[antecedent_key]
            if current_rule.consequent != new_rule.consequent and degree > current_degree:
                rules[antecedent_key] = (new_rule, degree)
        else:
            rules[antecedent_key] = (new_rule, degree)

        rule_n += 1

    # Unpack the dictionary into the resulting rules
    unpacked_rules = []
    for i in rules:
        unpacked_rules.append(rules[i][0])

    print('Total possible rules {}, had {} conflicting rules, resulted in total of {} rules'.format( rule_n, rule_n - len(unpacked_rules), len(unpacked_rules)))

    return Rulebase(unpacked_rules)


def multiply(x,y):
    return x * y


# def rule_degree(rule, input_data, output_data):
#     n = len(input_data)
#     degree = 0
#     for i in range(n):

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
    input_variables, output_var = get_variables(data)
    rulebase = generate_rules(data, input_variables, output_var)

    # print(rulebase)
    # print(data.head())
    # get_variables(data)
    # print(data.shape)

