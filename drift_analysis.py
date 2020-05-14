from latex_results import *
from load_data import get_measures_mean_across_experiments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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


def plot_calibrated_percentiles(measures, extract_data_fn, axs, percent, n_classes, isomerous=True):
    measures = sorted(measures, key=lambda m: m[0])  # Sort by classifier name
    # No matter how the data is structured, `extracted_data_fn` should always return
    # a list of Measure items
    extracted_data = extract_data_fn(measures)
    data = __extract_percentiles_data(extracted_data, percent, isomerous)

    clf_keys = list(filter(lambda k: 'calib' in k.lower(), data.keys()))
    x = list(range(1, int(100 // (percent * 100)) + 1))

    for j in range(2):
        for clf_name in clf_keys:
            ax = axs[j]
            data_clf = data[clf_name]
            err = 'BS' if j == 0 else 'CE'
            ax.plot(x, data_clf[err], '-o', label=clf_name)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_xticks(x)
            if j == 0:
                ax.set_ylabel(f'{n_classes} classes')

    calib = {k: v for k, v in data.items() if k in clf_keys}
    # Transform from {'BS': [.x, .x, .x, .x] ...} to {'BS': {'0': .x, '1': .x, ...}, 'CE': ...}'
    collected = list(map(lambda l: dict((k, {f'{j}': val for j, val in enumerate(v)}) for k, v in l.items()), calib.values()))
    df = pd.io.json.json_normalize(collected).mean().values
    axs[0].plot(x, df[:4], 'k--', label='Average of learners')
    axs[1].plot(x, df[4:8], 'k--', label='Average of learners')


def __extract_percentiles_data(extracted_data, percent, isomerous):
    measure_labels = ('BS', 'CE', 'Drift')
    data = {}
    for clf_name, extracted in extracted_data:
        data[clf_name] = {k: [] for k in measure_labels}
        # Sort by NAE
        sorted_data = sorted(extracted, key=lambda k: normalized_absolute_error(k.train_priors[1][0], k.test_priors[1][0]))
        for i, percentile in enumerate(grouper(sorted_data, int(len(sorted_data) * percent))):
            measure_mean = get_measures_mean_across_experiments(percentile)
            nae_mean = np.array([normalized_absolute_error(m.train_priors[1][0], m.test_priors[1][0]) for m in percentile]).mean()
            _, _, bs, em_bs, cal, em_cal, _, _ = get_latex_measures(measure_mean, isomerous)
            data[clf_name]['BS'].append(error_reduction_percentage(bs, em_bs))
            data[clf_name]['CE'].append(error_reduction_percentage(cal, em_cal))
            data[clf_name]['Drift'].append(nae_mean)
    return data


def table_bin_quartiles(measures, extract_data_fn, isomerous=True):
    measures = sorted(measures, key=lambda m: m[0])  # Sort by classifier name
    # No matter how the data is structured, `extracted_data_fn` should always return
    # a list of Measure items
    extracted_data = extract_data_fn(measures)
    data = {}
    quartile_labels = ('1st quartile', '2nd quartile', '3rd quartile', '4th quartile')
    idx_to_quartile = {i: k for i, k in enumerate(quartile_labels)}
    measure_labels = ('NAE', 'BS', 'CE', 'RE')
    for clf_name, extracted in extracted_data:
        data[clf_name] = {l: {} for l in quartile_labels}

        # Sort by NAE
        sorted_data = sorted(extracted, key=lambda k: normalized_absolute_error(k.train_priors[1][0], k.test_priors[1][0]))
        for i, quartile in enumerate(grouper(sorted_data, len(sorted_data) // 4)):
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
        yield template.substitute(
            measure=label,
            first_column=quartile_labels[0],
            second_column=quartile_labels[1],
            third_column=quartile_labels[2],
            fourth_column=quartile_labels[3],
            lr_nocalib=__values_to_latex(no_calib[LR_KEY].values(), label),
            rf_nocalib=__values_to_latex(no_calib[RF_KEY].values(), label),
            mnb_nocalib=__values_to_latex(no_calib[MNB_KEY].values(), label),
            avg_nocalib=" & ".join(f"{format_percentages(v)}\\%" for v in no_calib_averages[ind]),
            lr_calib=__values_to_latex(calib["Calibrated-" + LR_KEY].values(), label),
            rf_calib=__values_to_latex(calib["Calibrated-" + RF_KEY].values(), label),
            mnb_calib=__values_to_latex(calib["Calibrated-" + MNB_KEY].values(), label),
            svm_calib=__values_to_latex(calib[SVM_KEY].values(), label),
            avg_calib=" & ".join(f"{format_percentages(v)}\\%" for v in calib_averages[ind])
        )


def __values_to_latex(data, measure_label):
    return " & ".join(f"{format_percentages(quartile_data[measure_label])}\\%" for quartile_data in data)


def table_bin_10_percent(measures, extract_data_fn, isomerous=True):
    measures = sorted(measures, key=lambda m: m[0])  # Sort by classifier name
    # No matter how the data is structured, `extracted_data_fn` should always return
    # a list of Measure items
    extracted_data = extract_data_fn(measures)
    data = {}
    percent_labels = ('1st 10 percent', 'last 10 percent')
    idx_to_percent = {i: k for i, k in enumerate(percent_labels)}
    measure_labels = ('NAE', 'BS', 'CE', 'RE')
    for clf_name, extracted in extracted_data:
        data[clf_name] = {l: {} for l in percent_labels}

        # Sort by NAE
        sorted_data = sorted(extracted, key=lambda k: normalized_absolute_error(k.train_priors[1][0], k.test_priors[1][0]))
        ten_percents = sorted_data[:int(len(sorted_data) * 0.1)], sorted_data[int(len(sorted_data) * 0.9):]
        for i, ten_percent in enumerate(ten_percents):
            measure_mean = get_measures_mean_across_experiments(ten_percent)
            data[clf_name][idx_to_percent[i]] = dict(
                zip(
                    measure_labels,
                    [error_reduction_percentage(n[0], n[1]) for n in grouper(get_latex_measures(measure_mean, isomerous), 2)]
                )
            )

    calib = dict((k, v) for k, v in data.items() if 'Calibrated' in k)
    no_calib = dict((k, v) for k, v in data.items() if 'Calibrated' not in k)
    with open('shift_template_10.tex', 'r') as f:
        template = string.Template(f.read())

    calib_averages = pd.io.json.json_normalize(calib.values()).mean()
    no_calib_averages = pd.io.json.json_normalize(no_calib.values()).mean()

    for label in measure_labels:
        ind = [ind for ind in no_calib_averages.index if label in ind]
        yield template.substitute(
            measure=label,
            first_column=percent_labels[0],
            second_column=percent_labels[1],
            lr_nocalib=__values_to_latex(no_calib[LR_KEY].values(), label),
            rf_nocalib=__values_to_latex(no_calib[RF_KEY].values(), label),
            mnb_nocalib=__values_to_latex(no_calib[MNB_KEY].values(), label),
            avg_nocalib=" & ".join(f"{format_percentages(v)}\\%" for v in no_calib_averages[ind]),
            lr_calib=__values_to_latex(calib["Calibrated-" + LR_KEY].values(), label),
            rf_calib=__values_to_latex(calib["Calibrated-" + RF_KEY].values(), label),
            mnb_calib=__values_to_latex(calib["Calibrated-" + MNB_KEY].values(), label),
            svm_calib=__values_to_latex(calib[SVM_KEY].values(), label),
            avg_calib=" & ".join(f"{format_percentages(v)}\\%" for v in calib_averages[ind])
        )


def table_min_max_quartiles(measures, extract_data_fn):
    measures = sorted(measures, key=lambda m: m[0])  # Sort by classifier name
    # No matter how the data is structured, `extracted_data_fn` should always return
    # a list of Measure items
    extracted_data = extract_data_fn(measures)
    data = [{'min': [], 'max': []} for _ in range(4)]
    for clf_name, extracted in extracted_data:
        if 'calib' not in clf_name.lower():
            continue
        # Sort by NAE
        sorted_data = sorted(extracted, key=lambda k: normalized_absolute_error(k.train_priors[1][0], k.test_priors[1][0]))
        for i, percentile in enumerate(grouper(sorted_data, int(len(sorted_data) * 0.25))):
            percentile = list(filter(lambda p: p is not None, percentile))
            data[i]['min'].append(normalized_absolute_error(percentile[0].train_priors[1][0], percentile[0].test_priors[1][0]))
            data[i]['max'].append(normalized_absolute_error(percentile[-1].train_priors[1][0], percentile[-1].test_priors[1][0]))

    return list(map(lambda d: dict((k, sum(v) / len(v)) for k, v in d.items()), data))


def extract_binary_data(measures):
    groups = itertools.groupby(measures, key=lambda k: k[0])
    for clf_name, group in groups:
        yield clf_name, flatten(map(lambda k: k[1], group))


if __name__ == '__main__':
    quartiles_data = [[] for _ in range(5)]
    for n_classes in ("2", "5", "10", "20", "37"):
        measures_gen = load_measure_for_classifiers_pickles("rcv1", n_classes)
        if n_classes == "2":
            data = table_min_max_quartiles(measures_gen, extract_binary_data)
        else:
            data = table_min_max_quartiles(measures_gen, lambda m: m)
        for j, d in enumerate(data):
            quartiles_data[j].append(d)

    with open('quartiles_template.tex', 'r') as f:
        template = string.Template(f.read())

    print(template.substitute(
        first_quartile=" & ".join(f"{v:.3f}" for i in quartiles_data[0] for v in (i['min'], i['max'])),
        second_quartile=" & ".join(f"{v:.3f}" for i in quartiles_data[1] for v in (i['min'], i['max'])),
        third_quartile=" & ".join(f"{v:.3f}" for i in quartiles_data[2] for v in (i['min'], i['max'])),
        fourth_quartile=" & ".join(f"{v:.3f}" for i in quartiles_data[3] for v in (i['min'], i['max'])),
    ))
    # fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=False, figsize=(15, 22))
    # axs[0, 0].set_title('Reduction in Brier Score')
    # axs[0, 1].set_title('Reduction in Calibration Error')
    # axs[len(axs) - 1, 0].set_xlabel('Quartiles')
    # axs[len(axs) - 1, 1].set_xlabel('Quartiles')
    # for i, n_classes in enumerate(("2", "5", "10", "20", "37")):
    # # for i, n_classes in enumerate(("5",)):
    #     print(f"Computing measures and plotting for {n_classes} classes")
    #     measures_gen = load_measure_for_classifiers_pickles("rcv1", n_classes)
    #     if n_classes == '2':
    #         plot_calibrated_percentiles(measures_gen, extract_binary_data, [axs[i, 0], axs[i, 1]], 0.25, n_classes)
    #     else:
    #         plot_calibrated_percentiles(measures_gen, lambda m: m, [axs[i, 0], axs[i, 1]], 0.25, n_classes)
    #
    # axs[0, 1].legend(loc='lower right')
    # fig.tight_layout()
    # fig.savefig('quartiles.pdf')
