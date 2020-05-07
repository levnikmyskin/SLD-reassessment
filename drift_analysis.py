from latex_results import *
from load_data import get_measures_mean_across_experiments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import itertools


def grouper(iterable, n, fillvalue=None):
    # Collect data into fixed-length chunks or blocks
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def mean_absolute_error(train_pr, test_pr):
    return sum(abs(t1 - t2) for t1, t2 in zip(train_pr, test_pr)) / len(train_pr)


def plot_drift_mean_across_iterations(drift_dict):
    data = dict(map(lambda kv: (kv[0], np.mean(kv[1])), drift_dict.items()))
    plt.bar(data.keys(), data.values())
    plt.xlabel("Number of classes")
    plt.ylabel("Drift mean across 500 experiments")
    plt.show()


def plot_drift_for_iterations(drift_dict, class_, use_mae):
    class_data = drift_dict[class_]
    plt.figure(figsize=(8, 6))
    plt.ylim(top=1)
    plt.bar(np.arange(0, len(class_data)), class_data, width=3)
    plt.xlabel(f"Experiments (with {class_} classes)")
    plt.ylabel(f"Drift {'MAE' if use_mae else 'NAE'}")
    plt.savefig(f'{class_}cls_drift_{"MAE" if use_mae else "NAE"}', figsize=(8,6))
    # plt.show()


def plot_drift_binning(bins_dict, n_classes):
    x = [f"{k}-{k + .25}" for k in bins_dict.keys()]
    y = [len(v) for v in bins_dict.values()]

    plt.figure(figsize=(8, 6))
    plt.xlabel(f"Bin intervals ({n_classes} classes)")
    plt.ylabel("Count of bin items")
    plt.bar(x, y)
    plt.tight_layout()
    plt.show()


def table_bin_quartiles(measures, isomerous=True):
    measures = sorted(measures, key=lambda m: m[0])  # Sort by classifier name
    groups = itertools.groupby(measures, key=lambda m: m[0])  # Todo what's this? Not necessary
    data = {}
    quartile_labels = ('First quartile', 'Second quartile', 'Third quartile', 'Forth quartile')
    idx_to_quartile = {i: k for i, k in enumerate(quartile_labels)}
    measure_labels = ('NAE', 'Brier', 'Cal', 'Ref')
    for clf_name, group in groups:
        data[clf_name] = {l: {} for l in quartile_labels}

        # Sort by NAE
        sorted_group = next(map(
            lambda m:
            sorted(m[1], key=lambda k: normalized_absolute_error(k.train_priors[1][0], k.test_priors[1][0])),
            group
        ))
        for i, quartile in enumerate(grouper(sorted_group, len(sorted_group) // 4)):
            measure_mean = get_measures_mean_across_experiments(quartile)
            data[clf_name][idx_to_quartile[i]] = dict(
                zip(
                    measure_labels,
                    [error_reduction_percentage(n[0], n[1]) for n in grouper(get_latex_measures(measure_mean, isomerous), 2)]
                )
            )

    calib = dict((k, v) for k, v in data.items() if 'Calibrated' in k)
    no_calib = dict((k, v) for k, v in data.items() if 'Calibrated' not in k)
    with open('shift_template.tex', 'r') as f:
        template = string.Template(f.read())

    calib_averages = pd.io.json.json_normalize(calib.values()).mean()
    no_calib_averages = pd.io.json.json_normalize(no_calib.values()).mean()

    for label in measure_labels:
        ind = [ind for ind in no_calib_averages.index if label in ind]
        template.substitute(
            measure=label,
            first_column=quartile_labels[0],
            second_column=quartile_labels[1],
            third_column=quartile_labels[2],
            fourth_column=quartile_labels[3],
            lr_nocalib=__values_to_latex(no_calib[LR_KEY].values(), label),
            rf_nocalib=__values_to_latex(no_calib[RF_KEY].values(), label),
            mnb_nocalib=__values_to_latex(no_calib[MNB_KEY].values(), label),
            avg_nocalib=" & ".join(f"{format_percentages(v)}\\%" for v in no_calib_averages[ind]) + "\\\\",
            lr_calib=__values_to_latex(calib["Calibrated-" + LR_KEY].values(), label),
            rf_calib=__values_to_latex(calib["Calibrated-" + RF_KEY].values(), label),
            mnb_calib=__values_to_latex(calib["Calibrated-" + MNB_KEY].values(), label),
            svm_calib=__values_to_latex(calib[SVM_KEY].values(), label),
            avg_calib=" & ".join(f"{format_percentages(v)}\\%" for v in calib_averages[ind]) + "\\\\"
        )

    return data


def __values_to_latex(data, measure_label):
    s = ""
    for quartile_data in data:
        s += f"{format_percentages(quartile_data[measure_label])}\\% &"
    s += "\\\\"
    return s


def table_bin_10_percent(measures):
    pass


if __name__ == '__main__':
    n_classes = "37"
    measures_gen = load_measure_for_classifiers_pickles("rcv1", n_classes)
    bins_dict = binning_by_drift(measures_gen)
    plot_drift_binning(bins_dict, n_classes)
