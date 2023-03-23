import torch


def soft_voting(probabilities):
    # Take the average
    votes = sum(probabilities) / len(probabilities)
    return votes


def hard_voting(probabilities):
    # Take the max
    probabilities = torch.swapdims(torch.stack(probabilities), 0, 1)
    votes = torch.tensor(torch.argmax(probabilities, dim=1).float(), requires_grad=True)
    return votes
