from latex_results import load_measure_for_classifiers_pickles, normalized_absolute_error, binning_by_drift
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


def plot_drift_binning(bins_dict, n_classes):
    x = [f"{k}-{k + .25}" for k in bins_dict.keys()]
    y = [len(v) for v in bins_dict.values()]

    plt.figure(figsize=(8, 6))
    plt.xlabel(f"Bin intervals ({n_classes} classes)")
    plt.ylabel("Count of bin items")
    plt.bar(x, y)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    n_classes = "37"
    measures_gen = load_measure_for_classifiers_pickles("rcv1", n_classes)
    bins_dict = binning_by_drift(measures_gen)
    plot_drift_binning(bins_dict, n_classes)
