import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # for overflow
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


