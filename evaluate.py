import numpy as np
from dataset import *
from rule_generator import *

def validate_sys(reasoner, test_data, thresh):
    test_x, test_y = x_y_split(test_data)
    correct = 0
    result = np.zeros((len(test_x), 2))
    confusion_matrix = np.zeros((5,5))
    non_classified = 0

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

    percentage_correct = correct / len(test_data) * 100

    print(result, result.shape)
    print(confusion_matrix)
    print('Had {} test examples that could not be classified by the rules'.format(non_classified))
    print('Percentage correct: {}%'.format(percentage_correct))

    return percentage_correct

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