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
        if start <= x <= top:
            membership_value = (x - start) /(top - start)
        elif top < x <= end:
            membership_value = (end - x) /(end - top)
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
            return (x-self.start)/(self.left_top-self.start)
        if x >= self.left_top and x <= self.right_top:
            return 1
        if x >= self.right_top and x <= self.end:
            return (x-self.end)/(self.right_top-self.end)
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
            mf.name : mf.calculate_membership(x)
            for mf in self.mfs
        }

    def get_mf_by_name(self, name):
        for mf in self.mfs:
            if mf.name == name:
                return mf

class Input(Variable):
    def __init__(self, name, range, mfs):
        super().__init__(name, range, mfs)
        self.type = "input"

class Output(Variable):
    def __init__(self, name, range, mfs):
        super().__init__(name, range, mfs)
        self.type = "output"


class Rule:
    """Fuzzy rule class, initialized with an antecedent (list of strings),
    operator (string) and consequent (string).
    vb.: rule1 = Rule(1, ["low", "amazing"], "and", "low")"""
    def __init__(self, n, antecedent, operator, consequent):
        self.number = n
        self.antecedent = antecedent
        self.operator = operator
        self.consequent = consequent
        self.firing_strength = 0

    def calculate_firing_strength(self, datapoint, inputs):
        res = []
        for i in range(len(inputs)):
            res_dict = inputs[i].calculate_memberships(datapoint[i])
            res.append(res_dict[self.antecedent[i]])
        self.firing_strength = min(res)

        return self.firing_strength


class Rulebase:
    """The fuzzy rulebase collects all rules for the FLS, can
    calculate the firing strengths of its rules."""
    def __init__(self, rules):
        self.rules = rules

    def calculate_firing_strengths(self, datapoint, inputs):
        result = Counter()
        for i, rule in enumerate(self.rules):
            fs = rule.calculate_firing_strength(datapoint, inputs)
            consequent = rule.consequent
            if fs > result[consequent]:
                result[consequent] = fs
        return result
