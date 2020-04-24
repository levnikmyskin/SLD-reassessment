import itertools
import sys
import pickle
import numpy as np
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_rcv1
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from multiprocessing import Pool
from data_generation import randomly_modify_prevalences
from load_data import get_measures_from_singlehist_measures
from datetime import date

from em import em
logging.basicConfig(filename="computation.log", level=logging.INFO, format='%(levelname)s:%(message)s')


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


# def run_experiment(batch_name, tr_prevalences, te_prevalences, classifier, y_min=0, y_max=1.0):
#     name, clf = classifier
#     # fig, axs = plt.subplots(len(tr_prevalences), len(te_prevalences))
#     # fig.suptitle(name)
#     # fig.set_size_inches(20, 20)
#     # axs_iter = iter(axs.flat)
#     for tr_pr in tr_prevalences:
#         for te_pr in te_prevalences:
#             # ax = next(axs_iter)
#             pos_count = y.sum()
#             neg_count = len(y) - pos_count
#             tr_pos = int(pos_count * tr_pr / (tr_pr + te_pr))
#             te_pos = pos_count - tr_pos
#             tr_neg = int(tr_pos / tr_pr * (1 - tr_pr))
#             te_neg = int(te_pos / te_pr * (1 - te_pr))
#             if tr_neg + te_neg > neg_count:
#                 factor = neg_count / (tr_neg + te_neg)
#                 tr_neg = int(tr_neg * factor)
#                 te_neg = neg_count - tr_neg
#                 tr_pos = int(tr_pos * factor)
#                 te_pos = int(te_pos * factor)
#             tr_idx = list()
#             for i in range(len(y)):
#                 if y[i] == 1:
#                     tr_idx.append(i)
#                     if len(tr_idx) == tr_pos:
#                         break
#             for i in range(len(y)):
#                 if y[i] == 0:
#                     tr_idx.append(i)
#                     if len(tr_idx) == tr_pos + tr_neg:
#                         break
#             te_idx = list()
#             for i in range(len(y)):
#                 if y[i] == 1 and i not in tr_idx:
#                     te_idx.append(i)
#                     if len(te_idx) == te_pos:
#                         break
#             for i in range(len(y)):
#                 if y[i] == 0 and i not in tr_idx:
#                     te_idx.append(i)
#                     if len(te_idx) == te_pos + te_neg:
#                         break
#             X_tr = X[tr_idx]
#             y_tr = y[tr_idx]
#             X_te = X[te_idx]
#             y_te = y[te_idx]
#
#             history = em_experiment(clf, X_tr, y_tr, X_te, y_te, y_min, y_max)
#             with open(f'./pickles/{batch_name}_{name.replace(" ", "-")}_{round(tr_pr, 3)}_{round(te_pr, 3)}.pkl',
#                       'wb') as f:
#                 print("Saving file ", f.name)
#                 pickle.dump((history, tr_pr, te_pr), f)

def run_experiment(batch_name, cls, x_tr, y_tr, x_te, y_te):
    name, cls = cls
    history = em_experiment(cls, x_tr, y_tr, x_te, y_te)
    with open(f'./pickles/{name.replace(" ", "-")}_{batch_name}.pkl', 'wb') as f:
        print("Saving file ", f.name)
        pickle.dump(history, f)


def batch(batch_name, x_train, y_train, x_test, y_test):
    classifiers = [#('Multinomial Bayes', MultinomialNB()),
        #('Calibrated Multinomial Bayes', CalibratedClassifierCV(MultinomialNB(), ensemble=False)),
        # ('Linear SVM', SVC(probability=True, kernel='linear')),
        ('Calibrated Linear SVM', CalibratedClassifierCV(SVC(kernel='linear'), ensemble=False)),
        ('Logistic Regression', LogisticRegression()),
        ('Calibrated Logistic Regression', CalibratedClassifierCV(LogisticRegression(), ensemble=False)),
        ('Random Forest', RandomForestClassifier()),
        ('Calibrated Random Forest', CalibratedClassifierCV(RandomForestClassifier(), ensemble=False))]

    with Pool(11) as p:
        p.starmap(run_experiment, [(batch_name, cls, x_train, y_train, x_test, y_test) for cls in classifiers])


def generate_n_randomly_modified_prevalence(n, x_tr, y_tr, x_te, y_te, train_sample, test_sample):
    for i in range(n):
        yield randomly_modify_prevalences(x_tr, y_tr, train_sample), randomly_modify_prevalences(x_te, y_te,
                                                                                                 test_sample)


def take(n, iterable):
    # Return first n items of the iterable as a list
    return list(itertools.islice(iterable, n))


def flatten(list_of_lists):
    # Flatten one level of nesting
    return itertools.chain.from_iterable(list_of_lists)


def run_n_iterations(n, x_train, y_train, x_test, y_test, classifier_name, multiclass, dataset_name, pool, n_classes, class_name="", take_n=50):
    logging.info(f"Running {n} iterations for classifier {classifier_name}")
    gen = generate_n_randomly_modified_prevalence(n, x_train, y_train, x_test, y_test, 7000, 10000)
    measures = list()
    i = 0
    while data := take(take_n, gen):
        logging.info(f"Processing dataset: subsample {i}-{i+take_n}/{n}")

        # Here we receive a list of measures computed over the EM history for a number of `take_n` subsamples.
        # Once we exit from the while loop, we'll have the measures list of length n, where each element is a Measure
        # object. We can save these to disk, and later compute a mean with the appropriate function in load_data.py
        measures.append(pool.starmap(
            em_experiment,
            [(classifier_name, new_xtr, new_ytr, new_xte, new_yte, multiclass) for (new_xtr, new_ytr), (new_xte, new_yte) in
             data],
            take_n // 10
        ))
        i += take_n

    logging.info(f"Saving measures for classifier {classifier_name}")
    with open(f'./pickles/measures/measures_{n}_{dataset_name}{class_name}_{n_classes}_{classifier_name.replace(" ", "-")}_{date.today().strftime("%d-%m-%y")}.pkl', 'wb') as f:
        pickle.dump(measures, f)


def run_n_iterations_no_parallel(n, x_train, y_train, x_test, y_test, classifier, multiclass, dataset_name, n_classes, class_name=""):
    classifier_name, classifier = classifier
    logging.info(f"Running {n} iterations for classifier {classifier_name}")
    gen = generate_n_randomly_modified_prevalence(n, x_train, y_train, x_test, y_test, 7000, 10000)
    measures = list()

    for i, ((new_xtr, new_ytr), (new_xte, new_yte)) in enumerate(gen):
        logging.info(f"Processing subsample {i+1}-{n}")
        measures.append(em_experiment(classifier, new_xtr, new_ytr, new_xte, new_yte, multiclass))

    logging.info(f"Saving measures for classifier {classifier_name}")
    with open(f'./pickles/measures/measures_{n}_{dataset_name}{class_name}_{n_classes}_{classifier_name.replace(" ", "-")}_{date.today().strftime("%d-%m-%y")}.pkl', 'wb') as f:
        pickle.dump(measures, f)


def rcv1_dataset(index='GCAT'):
    dataset = fetch_rcv1()
    index_class = list(dataset.target_names).index(index)
    y = np.asarray(dataset.target[:, index_class].todense()).squeeze()
    x_train, x_test = dataset.data[:23149], dataset.data[23149:]
    y_train, y_test = y[:23149], y[23149:]
    return x_train, x_test, y_train, y_test, "rcv1"


def twentyng_dataset_by_class(top_class):
    train_set = fetch_20newsgroups_vectorized(subset='train')
    test_set = fetch_20newsgroups_vectorized(subset='test')
    indices = []
    for i, class_ in enumerate(train_set.target_names):
        if top_class == class_.split('.')[0]:
            indices.append(i)
    y_train = np.where(np.isin(train_set.target, indices), 1, 0)
    y_test = np.where(np.isin(test_set.target, indices), 1, 0)
    return train_set.data, test_set.data, y_train, y_test, "20ng"


def twentyng_dataset(n_classes=-1, seed=None):
    def to_n_classes(arr, labels):
        # Summation of boolean arrays works as a logical OR
        mask = np.array(sum(x for x in [arr == i for i in labels]))
        return np.where(mask, arr, np.zeros_like(arr))

    train_set = fetch_20newsgroups_vectorized(subset='train')
    test_set = fetch_20newsgroups_vectorized(subset='test')
    if n_classes < 2:
        return train_set.data, test_set.data, train_set.target, test_set.target, "20ng"

    # Given a number N of desired output classes, we take N-1 random ints between 0 and 19 (20ng has 20 classes).
    # These will be our output classes. We zero out all other classes in the target array, and keep the selected
    # classes only. Eg. in the binary case, N=2, we take N-1=1 class which will be our positives, whereas the 0s will
    # be our negatives.
    np.random.seed(seed)
    train_target = train_set.target
    test_target = test_set.target
    labels = np.random.choice(20, n_classes, replace=False)

    train_target = to_n_classes(train_target, labels)
    test_target = to_n_classes(test_target, labels)

    # We convert labels so that they are incremental ids (eg. 0, 1, 2) instead of keeping the original labels.
    # This is for convenience of function which will later process these labels
    for i, label in enumerate(sorted(labels)):
        train_target[train_target == label] = i
        test_target[test_target == label] = i

    return train_set.data, test_set.data, train_target, test_target, "20ng"


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
        LogisticRegression()
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
    N_CLASSES = int(sys.argv[1])
    class_name = sys.argv[2] if len(sys.argv) >= 3 else ""
    full_x_train, full_x_test, full_y_train, full_y_test, dataset_name = twentyng_dataset_by_class(class_name)
    # for classifier in classifiers:
    #     run_n_iterations_no_parallel(ITERATIONS_NUMBER, full_x_train, full_y_train, full_x_test, full_y_test, classifier, False,
    #                                  dataset_name, N_CLASSES, class_name)
    with Pool(11, maxtasksperchild=ITERATIONS_NUMBER // 10) as p:
        for classifier in classifiers:
            run_n_iterations(ITERATIONS_NUMBER, full_x_train, full_y_train, full_x_test, full_y_test, classifier, False,
                             dataset_name, p, N_CLASSES, class_name, 100)

    # TODO scartare classi con meno di 900. Per ogni classe rimasta prendiamo esattamente 900 documenti a random.
    # TODO subsampling come prima andando a prendere 1000 documenti di training e 1000 di test
    # TODO subsampling random va fatto su tutto il dataset, assicurarsi che training e test non collidano