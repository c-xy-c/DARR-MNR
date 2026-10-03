"""User-directed accuracy target, separate from model and training objective."""
MINIMUM_ACCURACY_PERCENT = 70.


def active_support_branch(report):
    return (report['changed_answers_vs_anchor'] > 0
            and report['changed_answers_vs_swapped_support'] > 0)


def validation_allows_test(report):
    if report['full']['total'] != 14000:
        raise RuntimeError('complete validation required before test')
    return (100 * report['full']['correct'] / report['full']['total']
            >= MINIMUM_ACCURACY_PERCENT and active_support_branch(report))
