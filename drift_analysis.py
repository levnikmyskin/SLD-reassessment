from latex_results import load_measure_for_classifiers_pickles, normalized_absolute_error
import numpy as np
import matplotlib.pyplot as plt


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


def plot_drift_binning(drift_dict, class_, use_mae):
    class_data = np.array(drift_dict[class_])
    bin_steps = np.arange(0, 1, 0.1)
    bins = []
    for step in bin_steps:
        bins.append(np.logical_and(class_data > step, class_data <= step + 0.1).sum())

    plt.figure(figsize=(8, 6))
    plt.xlabel(f"Bin intervals ({class_} classes, {'MAE' if use_mae else 'NAE'})")
    plt.ylabel(f"Count of bin items")
    plt.bar(list(map(lambda b: f"({b:.1f}-{b+0.1:.1f}]", bin_steps)), bins)
    plt.tight_layout()
    plt.savefig(f'{class_}cls_bindrift_{"MAE" if use_mae else "NAE"}')


if __name__ == '__main__':
    classes = ["2", "5", "10", "20"]

    drifts = dict()
    for cls in classes:
        unpickled_data = load_measure_for_classifiers_pickles('20ng', cls)
        internal_data = dict()
        clf, data = next(unpickled_data)

        drift_for_iteration = []
        for it in data:
            train_pr, test_pr = it.train_priors[1][0], it.test_priors[1][0]
            if type(train_pr) == np.float64:
                train_pr, test_pr = [train_pr], [test_pr]
            drift_for_iteration.append(normalized_absolute_error(train_pr, test_pr))

        drifts[cls] = drift_for_iteration

    # plot_drift_mean_across_iterations(drifts)
    # plot_drift_for_iterations(drifts, "20", use_mae=False)
    plot_drift_binning(drifts, "2", use_mae=False)
