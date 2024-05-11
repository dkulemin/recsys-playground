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


def f_beta_score_at_k(
    preds: Iterable,
    labels: Iterable,
    beta: float,
    k: int,
) -> float:
    """
    """
    precision = precision_at_k(preds, labels, k)
    recall = recall_at_k(preds, labels, k)
    return (
        (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        if precision * recall > 0 else 0.
    )


def f1_score_at_k(preds: Iterable, labels: Iterable, k: int) -> float:
    """
    """
    return f_beta_score_at_k(preds, labels, 1, k)
