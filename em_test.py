import pickle
import numpy as np
import logging
import random
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from multiprocessing import Pool
from data_generation import randomly_modify_prevalences
from load_data import get_measures_from_singlehist_measures
from datetime import date
from dataset_helpers import take, Rcv1Helper

from em import em
logging.basicConfig(filename="computation.log", level=logging.INFO, format='%(asctime)s:%(message)s')


def em_experiment(clf, X_tr, y_tr, X_te, y_te, multi_class=False):
    mlb = MultiLabelBinarizer()
    mlb.fit(np.expand_dims(np.hstack((y_tr, y_te)), 1))
    y_tr_bin = mlb.transform(np.expand_dims(y_tr, 1))
    y_te_bin = mlb.transform(np.expand_dims(y_te, 1))
    train_priors = np.mean(y_tr_bin, 0)
    test_priors = np.mean(y_te_bin, 0)

    clf = init_classifiers(clf)
    print("Fitting", clf)
    clf.fit(X_tr, y_tr)
    test_posteriors = clf.predict_proba(X_te)
    posteriors_test_priors = np.mean(test_posteriors, axis=0)

    print('train priors', train_priors, sep='\n')
    print('test priors', test_priors, sep='\n')
    print('posteriors mean', posteriors_test_priors, sep='\n')
    print()

    em_test_posteriors, em_test_priors, history = em(y_te, test_posteriors, train_priors, multi_class=multi_class)
    measures = get_measures_from_singlehist_measures(history)

    # print('Results')
    # print('prior from:   train test  post  em')
    # for i, (a, b, c, d) in enumerate(
    #         zip(train_priors, test_priors, posteriors_test_priors, em_test_priors)):
    #     print(f'{i:11d} - {a:3.3f} {b:3.3f} {c:3.3f} {d:3.3f}')

    return measures


def generate_n_randomly_modified_prevalence(n, x, y, train_sample, test_sample):
    for i in range(n):
        yield randomly_modify_prevalences(x, y, train_sample, test_sample)


def generate_n_randomly_prevalences_random_classes(n, rcv1_helper, n_classes, train_sample, test_sample):
    for i in range(n):
        classes = random.choices(list(rcv1_helper.hierarchical_single_label_indices_cached.keys()), k=n_classes)
        indices = np.concatenate(list(rcv1_helper.hierarchical_single_label_indices_cached[c] for c in classes))
        y_temp = rcv1_helper.target[indices]  # it seems like we cannot index indices, label_indices together
        x, y = rcv1_helper.data[indices], y_temp[:, rcv1_helper.label_indices(classes)]
        del y_temp
        yield randomly_modify_prevalences(x, y, train_sample, test_sample)


def run_n_iterations(n, rcv1_helper, classifier_name, multiclass, dataset_name, pool, n_classes, class_name="", take_n=50):
    logging.info(f"Running {n} iterations for classifier {classifier_name}")
    gen = generate_n_randomly_prevalences_random_classes(n, rcv1_helper, n_classes, 1000, 1000)
    measures = list()
    i = 0
    while data := take(take_n, gen):
        logging.info(f"Processing dataset: subsample {i}-{i+take_n}/{n}")

        # Here we receive a list of measures computed over the EM history for a number of `take_n` subsamples.
        # Once we exit from the while loop, we'll have the measures list of length n, where each element is a Measure
        # object. We can save these to disk, and later compute a mean with the appropriate function in load_data.py
        measures.append(pool.starmap(
            em_experiment,
            [(classifier_name, new_xtr, new_ytr, new_xte, new_yte, multiclass) for new_xtr, new_ytr, new_xte, new_yte in
             data],
            take_n // 10
        ))
        i += take_n

    logging.info(f"Saving measures for classifier {classifier_name}")
    with open(f'./pickles/measures_new_experiments/measures_{n}_{dataset_name}{class_name}_{n_classes}_{classifier_name.replace(" ", "-")}_{date.today().strftime("%d-%m-%y")}.pkl', 'wb') as f:
        pickle.dump(measures, f)


def init_classifiers(name):
    if name == 'Multinomial Bayes':
        return MultinomialNB()
    elif name == 'Calibrated Multinomial Bayes':
        return CalibratedClassifierCV(MultinomialNB(), ensemble=False)
    elif name == 'Calibrated Linear SVM':
        return CalibratedClassifierCV(SVC(kernel='linear'), ensemble=False)
    elif name == 'Random Forest':
        return RandomForestClassifier()
    elif name == 'Calibrated Random Forest':
        return CalibratedClassifierCV(RandomForestClassifier(), ensemble=False)
    elif name == 'Logistic Regression':
        return LogisticRegression()
    elif name == 'Calibrated Logistic Regression':
        return CalibratedClassifierCV(LogisticRegression(), ensemble=False)


if __name__ == '__main__':

    classifiers = [
        'Multinomial Bayes',
        'Calibrated Multinomial Bayes',
        'Calibrated Linear SVM',
        'Random Forest',
        'Calibrated Random Forest',
        'Logistic Regression',
        'Calibrated Logistic Regression'
    ]

    ITERATIONS_NUMBER = 500

    rcv1_helper = Rcv1Helper()
    with Pool(11, maxtasksperchild=ITERATIONS_NUMBER // 10) as p:
        for n_classes in [5, 10, 20, 37]:
            for classifier in classifiers:
                run_n_iterations(ITERATIONS_NUMBER, rcv1_helper, classifier, n_classes > 2, "rcv1", p, n_classes, "", 100)
