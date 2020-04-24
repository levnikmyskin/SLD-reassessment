import numpy as np
import itertools
import random


def take(n, iterable):
    # Return first n items of the iterable as a list
    return list(itertools.islice(iterable, n))


def flatten(list_of_lists):
    # Flatten one level of nesting
    return itertools.chain.from_iterable(list_of_lists)


def to_n_labels(arr: np.array, labels: {int}):
    """
    :param arr should be the Y array of our dataset. This MUST be a 1D array, single-label or binary.
    :param labels a set containing the indices of the labels we want to scale down our dataset to.
    Other labels will be zeroed-out in the output array.
    :return a 1D array of binary or single-label classes, eg. [1, 0, 0, 1, 0] or [1,2,0,2,1]
    """
    # Summation of boolean arrays works as a logical OR
    mask = np.array(sum(x for x in [arr == i for i in labels]))
    return np.where(mask, arr, np.zeros_like(arr))


def rcv1_single_label_dataset(n_classes: int, rcv1_helper, minimum_examples=2000):
    single_label_dict = dict(filter(lambda kv: kv[1].shape[0] >= minimum_examples,
                                    rcv1_helper.hierarchical_single_labels_indices()))

    class_indices = rcv1_helper.label_indices(random.choices(list(single_label_dict.keys()), k=n_classes))

    single_label_indices = list(flatten(single_label_dict.values()))
    data, target = rcv1_helper.data[single_label_indices], rcv1_helper.target[single_label_indices]
    return data, target[:, class_indices]


def rcv1_binary_dataset(rcv1_helper, minimum_examples=2000):
    """
    Returns a generator with all possible binary dataset for single label classes (i.e. 37 dataset)
    which have a number of examples >= `minimum_examples`
    :return a tuple with X, Y, RCV1 class name
    """
    single_label_dict = dict(filter(lambda kv: kv[1].shape[0] >= minimum_examples,
                                    rcv1_helper.hierarchical_single_labels_indices()))
    class_indices = rcv1_helper.label_indices(single_label_dict.keys())
    single_label_indices = list(flatten(single_label_dict.values()))
    data, target = rcv1_helper.data[single_label_indices], rcv1_helper.target[single_label_indices]

    del single_label_dict, single_label_indices
    for index in class_indices:
        yield data, np.asarray(target[:, index].todense()).squeeze(), rcv1_helper.target_names[index]
