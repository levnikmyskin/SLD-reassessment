from latex_results import load_measure_for_classifiers_pickles, flatten
from drift_analysis import extract_binary_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string


def iterations_mean(meas):
    return sum(len(m.em_priors[1]) for m in meas) / len(meas)


def exceeded_iters_percent(meas):
    return (sum(len(m.em_priors[1]) >= 999 for m in meas) / len(meas)) * 100


def convergence_on_targets_table(targets=["2", "5", "10", "20", "37"]):
    exceed_label = "Exceed iters. percent."
    exceed_percent_labels = [exceed_label + str(i) for i in range(len(targets))]
    data = {t: {} for t in flatten(zip(targets, exceed_percent_labels))}
    calib_data = {t: {} for t in flatten(zip(targets, exceed_percent_labels))}
    for i, target in enumerate(targets):
        measures = load_measure_for_classifiers_pickles("rcv1", target)
        if target == '2':
            measures = extract_binary_data(sorted(measures, key=lambda k: k[0]))
        for clf_name, measure in measures:
            measure = list(measure)
            if 'calibrated' in clf_name.lower():
                clf_name = "".join(c for c in clf_name.replace('Calibrated', '') if c.isupper())
                calib_data[target][clf_name] = iterations_mean(measure)
                calib_data[exceed_label + str(i)][clf_name] = exceeded_iters_percent(measure)
            else:
                clf_name = "".join(c for c in clf_name if c.isupper())
                data[target][clf_name] = iterations_mean(measure)
                data[exceed_label + str(i)][clf_name] = exceeded_iters_percent(measure)

    df = pd.DataFrame(data)
    df.loc['avg'] = df.mean(axis=0)
    df_calib = pd.DataFrame(calib_data)
    df_calib.loc['avg'] = df_calib.mean(axis=0)
    with open('iterations_template.tex', 'r') as f:
        template = string.Template(f.read())

    print(template.substitute(
        lr_data=" & ".join(f"{x:.2f}" for x in df.loc['LR']),
        mnb_data=" & ".join(f"{x:.2f}" for x in df.loc['MB']),
        rf_data=" & ".join(f"{x:.2f}" for x in df.loc['RF']),
        avg_data=" & ".join(f"{x:.2f}" for x in df.loc['avg']),
        svm_data=" & ".join(f"{x:.2f}" for x in df_calib.loc['LSVM']),
        lr_calib_data=" & ".join(f"{x:.2f}" for x in df_calib.loc['LR']),
        mnb_calib_data=" & ".join(f"{x:.2f}" for x in df_calib.loc['MB']),
        rf_calib_data=" & ".join(f"{x:.2f}" for x in df_calib.loc['RF']),
        avg_calib_data=" & ".join(f"{x:.2f}" for x in df_calib.loc['avg'])
    ))


def plot_convergence_on_targets(targets=["2", "5", "10", "20", "37"]):
    fig, axs = plt.subplots(len(targets), 1)
    for i, target in enumerate(targets):
        measures = load_measure_for_classifiers_pickles("rcv1", target)
        xs = []
        ys = []
        if target == '2':
            measures = extract_binary_data(sorted(measures, key=lambda k: k[0]))
        for clf_name, measure in measures:
            xs.append(clf_name)
            ys.append(iterations_mean(list(measure)))

        axs[i].bar(np.arange(len(xs)), ys)
        axs[i].set_xticks(np.arange(len(xs)))
        axs[i].set_xticklabels(xs, rotation=90)

    plt.show()


if __name__ == '__main__':
    convergence_on_targets_table()
