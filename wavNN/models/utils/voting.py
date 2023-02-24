from torch import nn


def soft_voting(probabilities):
    # Take the average, then the max
    probabilities_sum = sum(probabilities) / len(probabilities)
    return nn.Softmax(probabilities_sum)


def hard_voting(probabilities):
    # Take the max of the max.
    votes = [nn.Softmax(prob) for prob in probabilities]
    return nn.Softmax(votes)
