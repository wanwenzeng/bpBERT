import sklearn.metrics as skm
import logging
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import matplotlib
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from scipy.spatial.distance import jensenshannon

# Metric helpers

def average_dict(dict_data):
    avg_dict = {}

    keys = dict_data[next(iter(dict_data))].keys()

    for key in keys:
        if isinstance(dict_data[next(iter(dict_data))][key], dict):
            avg_dict[key] = {}
            sub_keys = dict_data[next(iter(dict_data))][key].keys()

            for sub_key in sub_keys:
                avg_dict[key][sub_key] = sum(d[key][sub_key] for d in dict_data.values()) / len(dict_data)
        else:
            avg_dict[key] = sum(d[key] for d in dict_data.values()) / len(dict_data)
    
    return avg_dict


def bin_counts_max(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = x[:, (binsize * i):(binsize * (i + 1)), :].max(1)
    return xout


def bin_counts_amb(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2])).astype(float)
    for i in range(outlen):
        iterval = x[:, (binsize * i):(binsize * (i + 1)), :]
        has_amb = np.any(iterval == -1, axis=1)
        has_peak = np.any(iterval == 1, axis=1)
        # if no peak and has_amb -> -1
        # if no peak and no has_amb -> 0
        # if peak -> 1
        xout[:, i, :] = (has_peak - (1 - has_peak) * has_amb).astype(float)
    return xout


def bin_counts_summary(x, binsize=2, fn=np.sum):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = np.apply_along_axis(fn, 1, x[:, (binsize * i):(binsize * (i + 1)), :])
    return xout

def permute_array(arr, axis=0):
    """Permute array along a certain axis

    Args:
      arr: numpy array
      axis: axis along which to permute the array
    """
    if axis == 0:
        return np.random.permutation(arr)
    else:
        return np.random.permutation(arr.swapaxes(0, axis)).swapaxes(0, axis)


def eval_profile(yt, yp,
                 pos_min_threshold=0.05,
                 neg_max_threshold=0.01,
                 required_min_pos_counts=None,
                 binsizes=[1, 2, 4, 10]):
    """
    Evaluate the profile in terms of auPR

    Args:
      yt: true profile (counts)
      yp: predicted profile (fractions)
      pos_min_threshold: fraction threshold above which the position is
         considered to be a positive
      neg_max_threshold: fraction threshold bellow which the position is
         considered to be a negative
      required_min_pos_counts: smallest number of reads the peak should be
         supported by. All regions where 0.05 of the total reads would be
         less than required_min_pos_counts are excluded
    """
    # The filtering
    # criterion assures that each position in the positive class is
    # supported by at least required_min_pos_counts  of reads

    print((pos_min_threshold, neg_max_threshold, neg_max_threshold))

    if required_min_pos_counts is not None:
        do_eval = yt.sum(axis=1).mean(axis=1) > required_min_pos_counts / pos_min_threshold
    else:
        do_eval = yt.sum(axis=1).mean(axis=1) > 50

    # make sure everything sums to one
    yp = yp / yp.sum(axis=1, keepdims=True)
    fracs = yt / yt.sum(axis=1, keepdims=True)

    yp_random = permute_array(permute_array(yp[do_eval], axis=1), axis=0)
    out = defaultdict(dict)
    for binsize in binsizes:
        is_peak = (fracs >= pos_min_threshold).astype(float)
        ambigous = (fracs < pos_min_threshold) & (fracs >= neg_max_threshold)
        is_peak[ambigous] = -1
        y_true = np.ravel(bin_counts_amb(is_peak[do_eval], binsize))

        imbalance = np.sum(y_true == 1) / np.sum(y_true >= 0)
        n_positives = np.sum(y_true == 1)
        n_ambigous = np.sum(y_true == -1)
        frac_ambigous = n_ambigous / y_true.size

        try:
            res = auprc(y_true,
                        np.ravel(bin_counts_max(yp[do_eval], binsize)))
            res_random = auprc(y_true,
                               np.ravel(bin_counts_max(yp_random, binsize)))
        except Exception:
            res = np.nan
            res_random = np.nan

        out['auprc'][f"binsize={binsize}"] = res
        out['random_auprc'][f"binsize={binsize}"] = res_random
        out['n_positives'][f"binsize={binsize}"] = n_positives
        out['frac_ambigous'][f"binsize={binsize}"] = frac_ambigous
        out['imbalance'][f"binsize={binsize}"] = imbalance

    for binsize in binsizes:
        js_divergence = jensenshannon(bin_counts_summary(yp[do_eval], binsize), bin_counts_summary(fracs[do_eval], binsize), axis=1) ** 2
        js_divergence
        js_divergence = np.mean(js_divergence)
        out['js_divergence'][f"binsize={binsize}"] = js_divergence

    return out


class PeakPredictionProfileMetric:

    def __init__(self, pos_min_threshold=0.05,
                 neg_max_threshold=0.01,
                 required_min_pos_counts=None,
                 binsizes=[1, 2, 5, 10]):

        self.pos_min_threshold = pos_min_threshold
        self.neg_max_threshold = neg_max_threshold
        self.required_min_pos_counts = required_min_pos_counts
        self.binsizes = binsizes

    def __call__(self, y_true, y_pred):
        out = eval_profile(y_true, y_pred,
                           pos_min_threshold=self.pos_min_threshold,
                           neg_max_threshold=self.neg_max_threshold,
                           required_min_pos_counts=self.required_min_pos_counts,
                           binsizes=self.binsizes)

        return out



class BootstrapMetric:
    def __init__(self, metric, n):
        """
        Args:
          metric: a function accepting (y_true and y_pred) and
             returning the evaluation result
          n: number of bootstrap samples to draw
        """
        self.metric = metric
        self.n = n

    def __call__(self, y_true, y_pred):
        outl = []
        for i in range(self.n):
            bsamples = (
                pd.Series(np.arange(len(y_true))).sample(frac=1, replace=True).values
            )
            outl.append(self.metric(y_true[bsamples], y_pred[bsamples]))
        return outl


class MetricsList:
    """Wraps a list of metrics into a single metric returning a list"""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return [metric(y_true, y_pred) for metric in self.metrics]


class MetricsDict:
    """Wraps a dictionary of metrics into a single metric returning a dictionary"""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return {k: metric(y_true, y_pred) for k, metric in self.metrics.items()}


class MetricsTupleList:
    """Wraps a dictionary of metrics into a single metric returning a dictionary"""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return [(k, metric(y_true, y_pred)) for k, metric in self.metrics]


class MetricsOrderedDict:
    """Wraps a OrderedDict/tuple list of metrics into a single metric
    returning an OrderedDict
    """

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return OrderedDict([(k, metric(y_true, y_pred)) for k, metric in self.metrics])


class MetricsMultiTask:
    """Run the same metric across multiple tasks
    """

    def __init__(self, metrics, task_names=None):
        self.metrics = metrics
        self.task_names = task_names

    def __call__(self, y_true, y_pred):
        n_tasks = y_true.shape[1]
        if self.task_names is None:
            self.task_names = [i for i in range(n_tasks)]
        else:
            assert len(self.task_names) == n_tasks
        return OrderedDict([(task, self.metrics(y_true[:, i], y_pred[:, i]))
                            for i, task in enumerate(self.task_names)])


class MetricsAggregated:

    def __init__(self,
                 metrics,
                 agg_fn={"mean": np.mean, "std": np.std},
                 prefix=""):
        self.metrics
        self.agg_fn = agg_fn
        self.prefix = prefix

    def __call__(self, y_true, y_pred):
        out = self.metrics(y_true, y_pred)
        m = np.array(list(out.values()))
        return {self.prefix + k: fn(m) for k, fn in self.agg_fn}


class MetricsConcise:

    def __init__(self, metrics):
        import concise
        self.metrics_dict = OrderedDict([(m, concise.eval_metrics.get(m))
                                         for m in metrics])

    def __call__(self, y_true, y_pred):
        return OrderedDict([(m, fn(y_true, y_pred))
                            for m, fn in self.metrics_dict.items()])


# -----------------------------
# Binary classification
# Metric helpers
MASK_VALUE = -1
# Binary classification


def _mask_nan(y_true, y_pred):
    mask_array = ~np.isnan(y_true)
    if np.any(np.isnan(y_pred)):
        print("WARNING: y_pred contains {0}/{1} np.nan values. removing them...".
              format(np.sum(np.isnan(y_pred)), y_pred.size))
        mask_array = np.logical_and(mask_array, ~np.isnan(y_pred))
    return y_true[mask_array], y_pred[mask_array]


def _mask_value(y_true, y_pred, mask=MASK_VALUE):
    mask_array = y_true != mask
    return y_true[mask_array], y_pred[mask_array]


def _mask_value_nan(y_true, y_pred, mask=MASK_VALUE):
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return _mask_value(y_true, y_pred, mask)


def n_positive(y_true, y_pred):
    return y_true.sum()


def n_negative(y_true, y_pred):
    return (1 - y_true).sum()


def frac_positive(y_true, y_pred):
    return y_true.mean()


def accuracy(y_true, y_pred, round=True):
    """Classification accuracy
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.accuracy_score(y_true, y_pred)


def auc(y_true, y_pred, round=True):
    """Area under the ROC curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)

    if round:
        y_true = y_true.round()
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return skm.roc_auc_score(y_true, y_pred)


def auprc(y_true, y_pred):
    """Area under the precision-recall curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    return skm.average_precision_score(y_true, y_pred)


def mcc(y_true, y_pred, round=True):
    """Matthews correlation coefficient
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.matthews_corrcoef(y_true, y_pred)


def f1(y_true, y_pred, round=True):
    """F1 score: `2 * (p * r) / (p + r)`, where p=precision and r=recall.
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.f1_score(y_true, y_pred)


def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))


classification_metrics = [
    ("auPR", auprc),
    ("auROC", auc),
    ("accuracy", accuracy),
    ("n_positive", n_positive),
    ("n_negative", n_negative),
    ("frac_positive", frac_positive),
]


class ClassificationMetrics:
    """All classification metrics
    """
    cls_metrics = classification_metrics

    def __init__(self):
        self.classification_metric = MetricsOrderedDict(self.cls_metrics)

    def __call__(self, y_true, y_pred):
        return self.classification_metric(y_true, y_pred)
# TODO - add gin macro for a standard set of classification and regession metrics


# --------------------------------------------
# Regression

def cor(y_true, y_pred):
    """Compute Pearson correlation coefficient.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]


def kendall(y_true, y_pred, nb_sample=100000):
    """Kendall's tau coefficient, Kendall rank correlation coefficient
    """
    from scipy.stats import kendalltau
    y_true, y_pred = _mask_nan(y_true, y_pred)
    if len(y_true) > nb_sample:
        idx = np.arange(len(y_true))
        np.random.shuffle(idx)
        idx = idx[:nb_sample]
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    return kendalltau(y_true, y_pred)[0]


def mad(y_true, y_pred):
    """Median absolute deviation
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root mean-squared error
    """
    return np.sqrt(mse(y_true, y_pred))


def rrmse(y_true, y_pred):
    """1 - rmse
    """
    return 1 - rmse(y_true, y_pred)


def mse(y_true, y_pred):
    """Mean squared error
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return ((y_true - y_pred) ** 2).mean(axis=None)


def ermse(y_true, y_pred):
    """Exponentiated root-mean-squared error
    """
    return 10**np.sqrt(mse(y_true, y_pred))


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    var_resid = np.var(y_true - y_pred)
    var_y_true = np.var(y_true)
    return 1 - var_resid / var_y_true


def pearsonr(y_true, y_pred):
    from scipy.stats import pearsonr
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return pearsonr(y_true, y_pred)[0]


def spearmanr(y_true, y_pred):
    from scipy.stats import spearmanr
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return spearmanr(y_true, y_pred)[0]


def pearson_spearman(yt, yp):
    return {"pearsonr": pearsonr(yt, yp),
            "spearmanr": spearmanr(yt, yp)}


regression_metrics = [
    ("mse", mse),
    ("var_explained", var_explained),
    ("pearsonr", pearsonr),  # pearson and spearman correlation
    ("spearmanr", spearmanr),
    ("mad", mad),  # median absolute deviation
]


class RegressionMetrics:
    """All classification metrics
    """
    cls_metrics = regression_metrics

    def __init__(self):
        self.regression_metric = MetricsOrderedDict(self.cls_metrics)

    def __call__(self, y_true, y_pred):
        # squeeze the last dimension
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = np.ravel(y_true)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = np.ravel(y_pred)

        return self.regression_metric(y_true, y_pred)


# available eval metrics --------------------------------------------


BINARY_CLASS = ["auc", "auprc", "accuracy", "tpr", "tnr", "f1", "mcc"]
CATEGORY_CLASS = ["cat_acc"]
REGRESSION = ["mse", "mad", "cor", "ermse", "var_explained"]

AVAILABLE = BINARY_CLASS + CATEGORY_CLASS + REGRESSION