import functools
import math
import numpy as np
from collections import defaultdict, Counter


class TriangularMF:
    def __init__(self, name, start, top, end):
        self.name = name
        self.start = start
        self.top = top
        self.end = end

    def calculate_membership(self, x):
        start = self.start
        top = self.top
        end = self.end
        if x == top:
            return 1
        if start <= x <= top:
            membership_value = (x - start) / (top - start)
        elif top < x <= end:
            membership_value = (end - x) / (end - top)
        else:
            membership_value = 0
        return membership_value


class TrapezoidalMF:
    """Trapezoidal fuzzy logic membership function class."""

    def __init__(self, name, start, left_top, right_top, end):
        self.name = name
        self.start = start
        self.left_top = left_top
        self.right_top = right_top
        self.end = end

    def calculate_membership(self, x):
        if x <= self.start and self.start != self.left_top:
            return 0
        if x <= self.start and self.start == self.left_top:
            return 1
        if x >= self.start and x <= self.left_top:
            return (x - self.start) / (self.left_top - self.start)
        if x >= self.left_top and x <= self.right_top:
            return 1
        if x >= self.right_top and x <= self.end:
            return (x - self.end) / (self.right_top - self.end)
        if x >= self.end and self.end != self.right_top:
            return 0
        if self.right_top == self.end and x >= self.end:
            return 1


class Variable:
    def __init__(self, name, range, mfs):
        self.name = name
        self.range = range
        self.mfs = mfs

    def calculate_memberships(self, x):
        return {
            mf.name: mf.calculate_membership(x)
            for mf in self.mfs
        }

    def get_mf_by_name(self, name):
        for mf in self.mfs:
            if mf.name == name:
                return mf

    def get_variable_name(self):
        return self.name


class Input(Variable):
    def __init__(self, name, range, mfs):
        super(Input, self).__init__(name, range, mfs)
        self.type = "input"


class Output(Variable):
    def __init__(self, name, range, mfs):
        super(Output, self).__init__(name, range, mfs)
        self.type = "output"


class Rule:
    """Fuzzy rule class, initialized with an antecedent (list of strings),
    operator (string) and consequent (string).
    vb.: rule1 = Rule(1, ["low", "amazing"], "and", "low")"""

    def __init__(self, n, antecedent, operator, consequent):
        self.number = n
        self.antecedent = antecedent
        self.operator = operator.upper()
        self.consequent = consequent
        self.firing_strength = 0

    def calculate_firing_strength(self, datapoint, inputs):
        if self.operator == 'PROBOR':
            result = []
            for i in range(len(inputs)):
                res_dict = inputs[i].calculate_memberships(datapoint[i])
                result.append(res_dict[self.antecedent[i]])

            firing_strength = functools.reduce(self.probor, result)
            self.firing_strength = firing_strength

        if self.operator == 'AND':
            result = []
            for i in range(len(inputs)):
                res_dict = inputs[i].calculate_memberships(datapoint[i])
                result.append(res_dict[self.antecedent[i]])
            self.firing_strength = min(result)

        if self.operator == 'OR':
            result = []
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
                '13. #51 (thal)'
                # '14. #58 (num)',
            ]
            for x in self.antecedent:
                for i, c in enumerate(columns):
                    if c in x:
                        res_dict = inputs[i].calculate_memberships(datapoint[i])
                        result.append(res_dict[x])
            self.firing_strength = max(result)

        return self.firing_strength

    def probor(self, x, y):
        if x == 0:
            x = 1
        if y == 0:
            y = 1

        return x + y - x * y

class Rulebase:
    """The fuzzy rulebase collects all rules for the FLS, can
    calculate the firing strengths of its rules."""

    def __init__(self, rules):
        self.rules = rules

    def __str__(self):
        return '\n'.join(['Rule {}: [{}] {} {}'.format(r.number, ', '.join(r.antecedent), r.operator, r.consequent) for r in self.rules])

    def calculate_firing_strengths(self, datapoint, inputs):
        result = Counter()
        for i, rule in enumerate(self.rules):
            fs = rule.calculate_firing_strength(datapoint, inputs)
            consequent = rule.consequent[0]
            if fs > result[consequent]:
                result[consequent] = fs
        return result



''' vb. call:
thinker = Reasoner(rulebase, inputs, output, 201)
datapoint = [100, 1]
print(round(thinker.inference(datapoint)))
'''
class Reasoner:
    def __init__(self, rulebase, inputs, output, n_points):
        self.rulebase = rulebase
        self.inputs = inputs
        self.output = output
        self.discretize = n_points

    def inference(self, datapoint):
        # 1. Calculate the highest firing strength found in the rules per
        # membership function of the output variable
        # looks like: {"low":0.5, "medium":0.25, "high":0}
        firing_strengths = self.rulebase.calculate_firing_strengths(datapoint, self.inputs)

        # 2. Aggregate and discretize
        # looks like: [(0.0, 1), (1.2437810945273631, 1), (2.4875621890547261, 1), (3.7313432835820892, 1), ...]
        input_value_pairs = self.aggregate(firing_strengths)

        # 3. Defuzzify
        # looks like a scalar
        crisp_output = self.defuzzify(input_value_pairs)

        return crisp_output

    def aggregate(self, firing_strengths):
        # First find where the aggregated area starts and ends
        mfs = []
        for key in firing_strengths:
            mf = self.output.get_mf_by_name(key)
            mfs.append((mf, firing_strengths[key]))
        res = []
        for x in np.linspace(0,4,self.discretize):
            mx = 0
            for (mf, fs) in mfs:
                tmp = mf.calculate_membership(x)
                if tmp > fs:
                    tmp = fs
                if tmp > mx:
                    mx = tmp
                    res.append((x,mx))
        return res

    def defuzzify(self, input_value_pairs):
        s1 = 0
        s2 = 0
        for (x,fs) in input_value_pairs:
            s1 += x*fs
            s2 += fs

        if s2 == 0 or s1 == 0:
            return None

        return s1/s2
