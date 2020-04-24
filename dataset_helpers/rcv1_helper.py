import re
import numpy as np
from sklearn.datasets import fetch_rcv1


class Rcv1Helper:

    def __init__(self, dataset=None):
        self.dataset = dataset if dataset is not None else fetch_rcv1()
        self.parent_hierarchy, self.children_hierarchy = self.__get_rcv1_hierarchy()

    def is_root(self, label):
        return len(self.parents(label)) == 0

    def is_leaf(self, label):
        return len(self.children(label)) == 0

    def parents(self, label) -> [str]:
        return self.parent_hierarchy[label]

    def children(self, label) -> {str}:
        return self.children_hierarchy[label]

    def label_indices(self, labels: [str]):
        target_names = list(self.dataset.target_names)
        return [target_names.index(label) for label in labels]

    def documents_per_class_rcv1(self) -> [(str, int)]:
        """
        Computes the number of documents for each RCV1 class.
        :return a list of tuples, where each tuple has the class name and the count of documents for that class
        """
        target_names = self.dataset.target_names
        return [(name, np.asarray(self.dataset.target[:, i].todense()).squeeze().sum())
                for i, name in enumerate(target_names)]

    def single_label_documents_per_class(self) -> [(str, int)]:
        """
        Computes the number of single-label multi-class documents for each RCV1 class.
        This does not take into account the RCV1 hierarchy and will basically only give counts for the 4 top
        classes. See `hierarchical_single_label_per_class`
        :return a list of tuples, where each tuple has the class name and the count of single-label documents
        for that class
        """
        indices, _ = np.where(np.sum(self.dataset.target, axis=1) == 1)
        single_labels = self.dataset.target[indices]
        target_names = self.dataset.target_names
        return [(name, np.asarray(single_labels[:, i].todense()).squeeze().sum())
                for i, name in enumerate(target_names)]

    def hierarchical_single_label_count_per_class(self, accumulate=False) -> [(str, int)]:
        """
        Computes the number of single-label multi-class documents for each RCV1 class. This takes into account
        the RCV1 hierarchy. If accumulate is set to True, each class will count both its examples and those of its
        children
        """
        target_names = list(self.dataset.target_names)
        single_labels = {}
        for label in self.parent_hierarchy.keys():
            indices = self.__single_label_indices(label, target_names)
            single_labels[label] = indices[0].shape[0]

        # Add to parents the count of their children
        if accumulate:
            sl_copy = single_labels.copy()
            for label, val in sl_copy.items():
                children = self.dataset.children(label)
                for child in children:
                    single_labels[label] += sl_copy[child]

        return [(name, val) for name, val in single_labels.items()]

    def hierarchical_single_labels_indices(self, label=None) -> [(str, np.array)]:
        """
        Returns the indices of `label` single-label multi-class document. If `label` is None, returns the indices
        for all labels.
        :return: a generator of dictionaries {label: indices}
        """
        target_names = list(self.dataset.target_names)
        if label:
            yield label, self.__single_label_indices(label, target_names)[0]
        else:
            for label in self.parent_hierarchy.keys():
                yield label, self.__single_label_indices(label, target_names)[0]

    def __single_label_indices(self, label, target_names):
        parents = self.parents(label)
        target_indices = [target_names.index(l) for l in [label] + parents]

        # First filter out all rows which have more or less than our desired number of labels
        # then take only those which have exactly our desired labels
        filtered_rows, _ = np.where(np.sum(self.dataset.target, axis=1) == len(target_indices))
        filtered_rows = self.dataset.target[filtered_rows]
        return np.where(np.asarray((filtered_rows[:, target_indices] == 1).todense()).all(axis=1))

    def __get_rcv1_hierarchy(self):
        def get_parents(label, hier, parents):
            if hier[label] == 'Root':
                return parents
            parents.append(hier[label])
            return get_parents(hier[label], hier, parents)

        parent_hier = {}
        pattern = re.compile(r'parent: (?P<parent>[\w\d]+)\s+child: (?P<child>[\w\d]+)')
        with open('./rcv1_hierarchy', 'r') as f:
            temp_hier = {}
            for line in f:
                m = pattern.search(line)
                if not m:
                    raise Exception("regex failed while parsing hierarchy!")
                if any(n == 'None' for n in [m.group('parent'), m.group('child')]):
                    continue

                temp_hier[m.group('child')] = m.group('parent')

            children_hier = {name: set() for name in temp_hier.keys()}
            for child in temp_hier.keys():
                parents_list = parent_hier.setdefault(child, [])
                parents = get_parents(child, temp_hier, parents_list)

                for parent in parents:
                    children = children_hier[parent]
                    children.add(child)

        return parent_hier, children_hier

    def __getattr__(self, item):
        # __getattr__ is invoked only if the item is not found the usual way, see
        # https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute
        return getattr(self.dataset, item)
