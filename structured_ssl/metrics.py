"""Balanced seven-layout RAVEN metrics used by selection and evaluation."""
from sspredrnet.data import CONFIGS


def summary(predictions, labels, configs):
    correct, total = [], []
    for index in range(7):
        mask = configs == index
        correct.append(int(((predictions == labels) & mask).sum()))
        total.append(int(mask.sum()))
    return summarize_counts(correct, total)


def summarize_counts(correct, total):
    if total != [2000] * 7:
        raise RuntimeError(f'incomplete evaluation split: {total}')
    accuracies = {name: 100 * n / t for name, n, t in zip(CONFIGS, correct, total)}
    return {'correct': sum(correct), 'total': sum(total), 'by_configuration': accuracies,
            'accuracy_macro': sum(accuracies.values()) / 7}
