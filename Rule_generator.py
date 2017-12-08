import math
import numpy as np
from collections import defaultdict, Counter
from dataset import get_clean_data
import fuzzy_functions as ff



dataset = get_clean_data()
print(dataset)



def Rule_gen(input_data, output_data, Variables):
	"""
		Generates a rule for each input and output pair
	:param input_data:
	:param output_data:
	:param Variables:
	:return:
	"""


	rule = ff.Rule(1, input_data, "and",  output_data )
	return rule,


def rule_degree(rule, input_data, output_data):
	n = len(input_data)
	degree = 0 
	for i in range(n):
