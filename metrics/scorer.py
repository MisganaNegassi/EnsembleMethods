#!/usr/bin/python3
# -*- coding: utf-8- -*-
import sklearn.metrics
import copy



def make_scorer(name, score_func, greater_is_better=True, needs_proba=False,
                needs_threshold=False, **kwargs):
    """Make a scorer from a performance metric or loss function.
    Factory inspired by scikit-learn which wraps scoring functions to be used in
    auto-sklearn. In difference to scikit-learn, auto-sklearn always needs to
    call ``predict_proba`` in order to have predictions on a seperate validation
    set to build ensembles with.
    Paramaters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.
    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.
    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification.
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.
    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(name, score_func, sign, kwargs)


# Standard regression scores
r2 = make_scorer('r2', sklearn.metrics.r2_score)
mean_squared_error = make_scorer('mean_squared_error',
                                 sklearn.metrics.mean_squared_error,
                                 greater_is_better=False)
mean_absolute_error = make_scorer('mean_absolute_error',
                                  sklearn.metrics.mean_absolute_error,
                                  greater_is_better=False)
median_absolute_error = make_scorer('median_absolute_error',
                                    sklearn.metrics.median_absolute_error,
                                    greater_is_better=False)

# Standard Classification Scores
accuracy = make_scorer('accuracy', sklearn.metrics.accuracy_score)
balanced_accuracy = make_scorer('balanced_accuracy',
                                classification_metrics.balanced_accuracy)
f1 = make_scorer('f1', sklearn.metrics.f1_score)

# Score functions that need decision values
roc_auc = make_scorer('roc_auc', sklearn.metrics.roc_auc_score,
                      greater_is_better=True, needs_threshold=True)
average_precision = make_scorer('average_precision',
                                sklearn.metrics.average_precision_score,
                                needs_threshold=True)
precision = make_scorer('precision', sklearn.metrics.precision_score)
recall = make_scorer('recall', sklearn.metrics.recall_score)

# Score function for probabilistic classification
log_loss = make_scorer('log_loss', sklearn.metrics.log_loss,
                       greater_is_better=False, needs_proba=True)
pac_score = make_scorer('pac_score', classification_metrics.pac_score,
                        greater_is_better=True, needs_proba=True)


CLASSIFICATION_METRICS = dict()

for scorer in [accuracy, balanced_accuracy, roc_auc, average_precision,
               log_loss, pac_score]:
    CLASSIFICATION_METRICS[scorer.name] = scorer

for name, metric in [('precision', sklearn.metrics.precision_score),
                     ('recall', sklearn.metrics.recall_score),
                     ('f1', sklearn.metrics.f1_score)]:
    globals()[name] = make_scorer(name, metric)
    CLASSIFICATION_METRICS[name] = globals()[name]
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        globals()[qualified_name] = make_scorer(qualified_name,
                                                partial(metric,
                                                        pos_label=None,
                                                        average=average))
        CLASSIFICATION_METRICS[qualified_name] = globals()[qualified_name]


def calculate_score(solution, prediction, task_type, metric,
                    all_scoring_functions=False):
    if task_type not in TASK_TYPES:
        raise NotImplementedError(task_type)

    if all_scoring_functions:
        score = dict()
        if task_type in REGRESSION_TASKS:
            # TODO put this into the regression metric itself
            cprediction = sanitize_array(prediction)
            metric_dict = copy.copy(REGRESSION_METRICS)
            metric_dict[metric.name] = metric
            for metric_ in REGRESSION_METRICS:
                func = REGRESSION_METRICS[metric_]
                score[func.name] = func(solution, cprediction)

        else:
            metric_dict = copy.copy(CLASSIFICATION_METRICS)
            metric_dict[metric.name] = metric
            for metric_ in metric_dict:
                func = CLASSIFICATION_METRICS[metric_]

                # TODO maybe annotate metrics to define which cases they can
                # handle?

                try:
                    score[func.name] = func(solution, prediction)
                except ValueError as e:
                    if e.args[0] == 'multiclass format is not supported':
                        continue
                    elif e.args[0] == 'Sample-based precision, recall, ' \
                                      'fscore is not meaningful outside ' \
                                      'multilabel classification. See the ' \
                                      'accuracy_score instead.':
                        continue
                    else:
                        raise e

    else:
        if task_type in REGRESSION_TASKS:
            # TODO put this into the regression metric itself
            cprediction = sanitize_array(prediction)
            score = metric(solution, cprediction)
        else:
            score = metric(solution, prediction)
