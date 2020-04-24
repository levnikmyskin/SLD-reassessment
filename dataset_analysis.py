from em_test import fetch_20newsgroups_vectorized, fetch_rcv1
import numpy as np
import pandas as pd
import re


class Rcv1Hierarchy:

    def __init__(self, dataset):
        self.dataset = dataset
        self.parent_hierarchy, self.children_hierarchy = self.__get_rcv1_hierarchy()

    def is_root(self, label):
        return len(self.parents(label)) == 0

    def is_leaf(self, label):
        return len(self.children(label)) == 0

    def parents(self, label):
        return self.parent_hierarchy[label]

    def children(self, label):
        return self.children_hierarchy[label]

    def __get_rcv1_hierarchy(self):
        def get_parents(label, hier, parents):
            if hier[label] == 'Root':
                return parents
            parents.append(hier[label])
            return get_parents(hier[label], hier, parents)

        parent_hier = {}
        pattern = re.compile(r'parent: (?P<parent>[\w\d]+)\s+child: (?P<child>[\w\d]+)')
        with open('rcv1_hierarchy', 'r') as f:
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


def document_per_class_rcv1(rcv1):
    target_names = rcv1.target_names
    return [(name, np.asarray(rcv1.target[:, i].todense()).squeeze().sum()) for i, name in enumerate(target_names)]


def document_per_class_20ng(twentyng):
    target_names = twentyng.target_names
    return [(name, (twentyng.target == i).sum()) for i, name in enumerate(target_names)]


def single_label_rcv1(rcv1):
    indices, _ = np.where(np.sum(rcv1.target, axis=1) == 1)
    single_labels = rcv1.target[indices]
    target_names = rcv1.target_names
    return [(name, np.asarray(single_labels[:, i].todense()).squeeze().sum()) for i, name in enumerate(target_names)], \
           single_labels.shape[0]


def single_label_rcv1_hierarchy(rcv1):
    hierarchy = Rcv1Hierarchy(rcv1)
    target_names = list(rcv1.target_names)
    single_labels = {}
    for label in hierarchy.parent_hierarchy.keys():
        parents = hierarchy.parents(label)
        target_indices = [target_names.index(l) for l in [label] + parents]

        # First filter out all rows which have more or less than our desired number of labels
        # then take only those which have exactly our desired labels
        filtered_rows, _ = np.where(np.sum(rcv1.target, axis=1) == len(target_indices))
        filtered_rows = rcv1.target[filtered_rows]
        indices = np.where(np.asarray((filtered_rows[:, target_indices] == 1).todense()).all(axis=1))
        single_labels[label] = indices[0].shape[0]

    # Add to parents the count of their children
    sl_copy = single_labels.copy()
    for label, val in sl_copy.items():
        if hierarchy.is_root(label):
            children = hierarchy.children(label)
            for child in children:
                single_labels[label] += sl_copy[child]

    return [(name, val) for name, val in single_labels.items()]


def save_csv_whole_datasets():
    rcv1 = fetch_rcv1()
    twentyng = fetch_20newsgroups_vectorized(subset='all')

    dpc_rcv1 = document_per_class_rcv1(rcv1)
    dpc_20ng = document_per_class_20ng(twentyng)

    df_rcv1 = pd.DataFrame(dpc_rcv1, columns=['Class', 'Count'])
    df_rcv1['Prevalence'] = df_rcv1['Count'].apply(lambda x: x / rcv1.target.shape[0])

    df_20ng = pd.DataFrame(dpc_20ng, columns=['Class', 'Count'])
    df_20ng['Prevalence'] = df_20ng['Count'].apply(lambda x: x / twentyng.target.shape[0])

    print(f"Dataframe for rcv1: {df_rcv1}\nDataframe for 20ng: {df_20ng}")
    print("Saving to csv")
    df_rcv1.to_csv('./rcv1.tsv', sep='\t')
    df_20ng.to_csv('./20nc.tsv', sep='\t')


def save_csv_rcv1_single_label():
    rcv1 = fetch_rcv1()
    single_labels, total_elems = single_label_rcv1(rcv1)
    df = pd.DataFrame(single_labels, columns=['Class', 'Count'])
    df['Prevalence wrt. dataset'] = df['Count'].apply(lambda x: x / rcv1.target.shape[0])
    df['Prevalence wrt. S.L. subset'] = df['Count'].apply(lambda x: x / total_elems)

    print(f"Dataframe for single-label rcv1: {df}")
    print("Saving to csv")
    df.to_csv('./rcv1_sl.tsv', sep='\t')


def save_csv_rcv1_single_label_hierarchy():
    rcv1 = fetch_rcv1()
    single_labels = single_label_rcv1_hierarchy(rcv1)
    subset_total_elems = sum(v for k, v in single_labels)
    df = pd.DataFrame(single_labels, columns=['Class', 'Count'])
    df['Prevalence wrt. dataset'] = df['Count'].apply(lambda x: x / rcv1.target.shape[0])
    df['Prevalence wrt. S.L. subset'] = df['Count'].apply(lambda x: x / subset_total_elems)

    print(f"Dataframe for single-label rcv1: {df}")
    print("Saving to csv")
    df.to_csv('./rcv1_sl_hierarchy.tsv', sep='\t')


if __name__ == '__main__':
    save_csv_rcv1_single_label_hierarchy()
