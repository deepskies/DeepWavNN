from torch import nn
from torch import tensor, stack


def soft_voting(probabilities):
    # Take the average
    votes = sum(probabilities) / len(probabilities)
    return votes


def hard_voting(probabilities):
    # Take the max
    probabilities = stack(probabilities).mT
    votes = nn.functional.softmax(probabilities, dim=0)
    return votes
