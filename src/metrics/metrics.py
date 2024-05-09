from typing import Iterable


def precision_at_k(preds: Iterable, labels: Iterable, k: int) -> float:
    """
    Top k precision score.

    This metric computes the number of times where the correct label
    is among the top k labels predicted.

    :param preds: ranked predictions
    :param labels: relevant labels
    :returns: value in range from 0.0 to 1.0
    """
    return len([i for i in preds[:k] if i in labels]) / k if k > 0 else 0.


def recall_at_k(preds: Iterable, labels: Iterable, k: int) -> float:
    """
    Top k recall score.

    This metric computes the number of times where top k relevant
    predictions among all relevant labels.

    :param preds: ranked predictions
    :param labels: relevant labels
    :returns: value in range from 0.0 to 1.0
    """
    return (
        len([i for i in preds[:k] if i in labels]) / len(labels)
        if labels else 0.
    )
