import itertools
import pickle
import re
import os
from multiprocessing import Pool

import numpy as np
from collections import namedtuple
from em import History, soft_accuracy, MeasureSingleHistory
from metrics import smoothmacroF1, isomerous_brier_decomposition, isometric_brier_decomposition
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import brier_score_loss

LoadedData = namedtuple('LoadedData', ('file_name', 'clf_name', 'dataset', 'it', 'date'))
Measures = namedtuple('Measures', ('soft_acc', 'em_soft_acc', 'soft_f1', 'em_soft_f1', 'abs_errors', 'em_abs_errors',
                                   'test_priors', 'train_priors', 'predict_priors', 'em_priors', 'brier', 'em_brier',
                                   'isometric_em_ref_loss', 'isometric_em_cal_loss',
                                   'isomerous_em_ref_loss', 'isomerous_em_cal_loss'))
labels_dict = dict(soft_acc='Soft accuracy', em_soft_acc='EM Soft acc.', soft_f1='Soft F1', em_soft_f1='EM soft F1',
                   abs_errors='Abs. err.', em_abs_errors='EM Abs. err.', test_priors='Test pr.',
                   train_priors='Train pr.',
                   predict_priors='Predict pr.', em_priors='EM pr.', brier='Brier', em_brier='EM brier',
                   isometric_em_ref_loss='Isomet. EM refin. loss', isometric_em_cal_loss='Isomet. EM cal. loss',
                   isomerous_em_ref_loss='Isomer. EM refin. loss', isomerous_em_cal_loss="Isomer. EM Cal. loss")

jupyter_measures = dict(nae="NAE", bs="Brier score", ce="Calibration error", re="Refinement error")


def load_em_history() -> [LoadedData]:
    pattern = re.compile(r'(?P<clf_name>([a-zA-Z\-]+))_(?P<dataset>.+)_(?P<it>\d+)_(?P<date>\d{2}-\d{2}-\d{2})')
    files = os.listdir('./pickles/')
    for f in files:
        m = pattern.match(f)
        if not m:
            print(f"Something wrong happened with file {f}. Data will not be loaded for this file.")
            continue
        else:
            d = m.groupdict()
            yield LoadedData(f"./pickles/{f}", d['clf_name'], d.get('dataset'), int(d['it']), d.get('date'))


def get_measures_from_history(history: [History], multi_class=False) -> Measures:
    y = history[0].y
    y_bin = MultiLabelBinarizer().fit_transform(np.expand_dims(y, 1))

    soft_acc = [soft_accuracy(y, history[0].posteriors)] * len(history)
    em_soft_acc = [soft_accuracy(y, elem.posteriors) for elem in history]
    f1 = [smoothmacroF1(y_bin, history[0].posteriors)] * len(history)
    em_soft_f1 = [smoothmacroF1(y_bin, elem.posteriors) for elem in history]

    if multi_class:
        test_priors = np.mean(y_bin, 0)
        abs_errors = [abs(test_priors - history[0].priors)] * len(history)
        em_abs_errors = [abs(test_priors - elem.priors) for elem in history]
        em_priors = [np.mean(elem.priors, 0) for elem in history]
        train_priors = [np.mean(history[0].priors, 0)] * len(history)
        predict_priors = [np.mean(history[0].posteriors, 0)] * len(history)
        brier = [0] * len(history)
        em_brier = brier
    else:
        test_priors = np.mean(y_bin, 0)[1]
        abs_errors = [abs(test_priors - history[0].priors[1])] * len(history)
        em_abs_errors = [abs(test_priors - elem.priors[1]) for elem in history]
        em_priors = [elem.priors[1] for elem in history]
        train_priors = [history[0].priors[1]] * len(history)
        predict_priors = [np.mean(history[0].posteriors[:, 1])] * len(history)
        brier = [brier_score_loss(y, history[0].posteriors[:, 1])] * len(history)
        em_brier = [brier_score_loss(y, elem.posteriors[:, 1]) for elem in history]

    isometric_em_brier_decomp = [isometric_brier_decomposition(y, elem.posteriors) for elem in history]
    isometric_em_cal_loss = list(map(lambda el: el[0], isometric_em_brier_decomp))
    isometric_em_ref_loss = list(map(lambda el: el[1], isometric_em_brier_decomp))

    isomerous_em_brier_decomp = [isomerous_brier_decomposition(y, elem.posteriors) for elem in history]
    isomerous_em_cal_loss = list(map(lambda el: el[0], isomerous_em_brier_decomp))
    isomerous_em_ref_loss = list(map(lambda el: el[1], isomerous_em_brier_decomp))

    return Measures(
        ('Soft accuracy', soft_acc),
        ('EM soft acc.', em_soft_acc),
        ('Soft F1', f1),
        ('EM soft F1', em_soft_f1),
        ('Abs. errs.', abs_errors),
        ('EM abs. errs.', em_abs_errors),
        ('Test pr.', [test_priors] * len(history)),
        ('Train pr.', train_priors),
        ('Predict pr.', predict_priors),
        ('EM pr.', em_priors),
        ('brier', brier),
        ('em_brier', em_brier),
        ('Isomet. EM refin. loss', isometric_em_ref_loss),
        ('Isomet. EM cal. loss', isometric_em_cal_loss),
        ('Isomer. EM refin. loss', isomerous_em_ref_loss),
        ('Isomer. EM Cal. loss', isomerous_em_cal_loss)
    )


def get_measures_from_singlehist_measures(measures: [MeasureSingleHistory]) -> Measures:
    return Measures(
        ('Soft accuracy', [measures[0].soft_acc] * len(measures)),
        ('EM soft acc.', [m.soft_acc for m in measures]),
        ('Soft F1', [measures[0].soft_f1] * len(measures)),
        ('EM soft F1', [m.soft_f1 for m in measures]),
        ('Abs. errs.', [measures[0].abs_errors] * len(measures)),
        ('EM abs. errs.', [m.abs_errors for m in measures]),
        ('Test pr.', [measures[0].test_priors] * len(measures)),
        ('Train pr.', [measures[0].train_priors] * len(measures)),
        ('Predict pr.', [measures[0].predict_priors] * len(measures)),
        ('EM pr.', [m.predict_priors for m in measures]),
        ('brier', [measures[0].brier] * len(measures)),
        ('em_brier', [m.brier for m in measures]),
        ('Isomet. EM refin. loss', [m.isometric_ref_loss for m in measures]),
        ('Isomet. EM cal. loss', [m.isometric_cal_loss for m in measures]),
        ('Isomer. EM refin. loss', [m.isomerous_ref_loss for m in measures]),
        ('Isomer. EM Cal. loss', [m.isomerous_cal_loss for m in measures])
    )


def get_measures_mean_across_experiments(experiment_measures: [Measures], filters: [str] = None) -> Measures:
    def pad_data(measures, field):
        padded = list()
        for elem in measures:
            elem_data = np.array(elem.__getattribute__(field)[1])

            pad_width = (0, max_len - elem_data.shape[0])
            if len(elem_data.shape) > 1:
                # If we have multiclass data, we need to pad on the first axis only
                pad = np.pad(elem_data, (pad_width, (0, 0)), mode='edge')
            else:
                pad = np.pad(elem_data, pad_width, mode='edge')
            padded.append(pad)
        return np.array(padded)

    max_len = max(len(elem.em_brier[1]) for elem in experiment_measures)

    out = list()
    for field in Measures._fields:
        if filters is None or field in filters:
            out.append((labels_dict[field], pad_data(experiment_measures, field).mean(axis=0)))
        else:
            out.append(None)

    return Measures._make(out)


def get_measures_for_all_iterations(loaded_data: LoadedData, multi_class=False):
    with open(loaded_data.file_name, 'rb') as f:
        return get_measures_from_history(pickle.load(f), multi_class)


if __name__ == '__main__':
    from datetime import date

    today = date.today().strftime("%d-%m-%y")
    pattn = re.compile(r'rcv1_\d+_' + today)
    em_hist = load_em_history()
    data = filter(lambda el: pattn.search(el.file_name), sorted(em_hist, key=lambda el: el.clf_name))
    groups = itertools.groupby(data, key=lambda el: el.clf_name)
    results = list()
    with Pool(11) as p:
        for group in groups:
            print(f"Processing group for classifier: {group[0]}")
            results.append((group[0], p.starmap(get_measures_for_all_iterations, [(el, True) for el in group[1]])))

    with open(f'./pickles/measures_all_em_rcv1_{date.today().strftime("%d-%m-%y")}.pkl', 'wb') as f:
        pickle.dump(results, f)
