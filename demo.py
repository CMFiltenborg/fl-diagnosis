from dataset import get_clean_data
import numpy as np
import pandas as pd
from fuzzy_functions import *
from membership import get_variables
from rule_generator import generate_rules

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

    df = pd.DataFrame(data, columns=columns)
    optimization_function = np.average
    best_columns = ['1. #3 (age)', '2. #4 (sex)', '3. #9 (cp)', '14. #58 (num)']
    df = df[best_columns]
    data = df.as_matrix()

    inputs, output = get_variables(df, {})  # get input and output variables and their memebership functions
    rulebase = generate_rules(df, inputs, output)  # generate rules
    reasoner = Reasoner(rulebase, inputs, output, 1000) # make a Reasoner object to initialize the whole system.


    example = data[100]

    output = reasoner.inference(example)
    print('\n')
    print(rulebase)
    sex = 'man' if example[1] == 1.0 else 'woman'
    print('\n')
    print('Example: Age:{}, Sex:{}, Chest-pain:{}'.format(example[0], sex, example[2]))
    print('\n')
    print('Classification: {}'.format(output))
