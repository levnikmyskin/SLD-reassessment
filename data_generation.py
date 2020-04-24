import numpy as np

"""
We want to subsample a given dataset D with a given prevalence p so that the subsample will have a random prevalence p'.
In order to do this we do the following:
- First, we create a number c of groups (we'll call the set of groups G) for every class C, each containing the examples
    from D pertaining to c;
- Secondly, we generate a vector V of random probabilities with length |C|, such that its values sum to 1:
    each element of this vector will represent the prevalence of a class c in C;
- Finally, we draw samples from G with probabilities coming from V. 

This works both for multi and single class datasets.
"""


def randomly_modify_prevalences(x: np.array, y: np.array, sample_length: int) -> (np.array, np.array):
    # Step one: generate groups
    labels = set(y)
    groups = [(y == label).nonzero()[0] for label in labels]

    # Step two: generate the vector of random probabilities
    rand_num = np.random.randint(1, 100, len(labels))
    rand_prob_vector = rand_num / rand_num.sum()

    # For details on what's happening here, see
    # https://stackoverflow.com/questions/44613347/multidimensional-array-for-random-choice-in-numpy
    lens = np.array([el.shape[0] for el in groups])
    new_arr = np.concatenate(groups)
    new_rand_vec = np.repeat(np.divide(rand_prob_vector, lens), lens)
    new_sample = np.random.default_rng().choice(new_arr, size=sample_length, p=new_rand_vec, replace=False)

    return x[new_sample], y[new_sample]


def subsample_dataset_random_prev(dataset, sample_length: int) -> ((np.array, np.array), (np.array, np.array)):
    x, y = dataset.data, dataset.target
    split = y.shape[0] // 2
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    return randomly_modify_prevalences(x_train, y_train, sample_length), \
           randomly_modify_prevalences(x_test, y_test, sample_length)


if __name__ == '__main__':
    from sklearn.datasets import fetch_20newsgroups_vectorized
    dataset = fetch_20newsgroups_vectorized()
    subsample_dataset_random_prev(dataset, 3000)
