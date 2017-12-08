from dataset import get_clean_data
from fuzzy_functions import *

data = get_clean_data()
print(data)


# feature_vec: vector of values for one feature
# n: number of memberships for the feature
# names: list of length n with names for the membership functions
def make_membership(feature_vec, n, names):
    r = [min(feature_vec), max(feature_vec)]
    d = (r[1]-r[0])/(n-1)
    for i in range(n):
        # make membership func
