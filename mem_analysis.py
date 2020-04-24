from em_test import *
from memory_profiler import profile


@profile
def run_n_iterations(n, x_train, y_train, x_test, y_test, classifier, multiclass, queue, nproc=11, take_n=50):
    classifier_name, classifier = classifier
    print(f"Running {n} iterations for classifier {classifier_name}")
    gen = generate_n_randomly_modified_prevalence(n, x_train, y_train, x_test, y_test, 7000, 10000)
    measures = list()
    for i, data in enumerate(gen):
        (new_xtr, new_ytr), (new_xte, new_yte) = data
        print(f"Processing {i}th dataset")
        hist = em_experiment(classifier, new_xtr, new_ytr, new_xte, new_yte, multiclass)
        measures.append(get_measures_from_singlehist_measures(hist))

    return classifier_name, measures


@profile
def test():
    full_x_train, full_x_test, full_y_train, full_y_test, dataset_name = twentyng_dataset()
    ITERATIONS_NUMBER = 5
    final_measures = list()
    classifier = ('Multinomial Bayes', MultinomialNB())
    final_measures.append(
        run_n_iterations(ITERATIONS_NUMBER, full_x_train, full_y_train, full_x_test, full_y_test, classifier, True,
                         None, 0, 0))


if __name__ == '__main__':
    test()
