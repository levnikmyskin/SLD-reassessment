from em_test import fetch_20newsgroups_vectorized
import pandas as pd
from dataset_helpers import Rcv1Helper


def document_per_class_20ng(twentyng):
    target_names = twentyng.target_names
    return [(name, (twentyng.target == i).sum()) for i, name in enumerate(target_names)]


def save_csv_whole_datasets():
    rcv1 = Rcv1Helper()
    twentyng = fetch_20newsgroups_vectorized(subset='all')

    dpc_rcv1 = rcv1.documents_per_class_rcv1()
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
    rcv1 = Rcv1Helper()
    single_labels = rcv1.single_label_documents_per_class()
    total_elems = sum(v for k,v in single_labels)
    df = pd.DataFrame(single_labels, columns=['Class', 'Count'])
    df['Prevalence wrt. dataset'] = df['Count'].apply(lambda x: x / rcv1.target.shape[0])
    df['Prevalence wrt. S.L. subset'] = df['Count'].apply(lambda x: x / total_elems)

    print(f"Dataframe for single-label rcv1: {df}")
    print("Saving to csv")
    df.to_csv('./rcv1_sl.tsv', sep='\t')


def save_csv_rcv1_single_label_hierarchy():
    rcv1 = Rcv1Helper()
    single_labels = rcv1.hierarchical_single_label_count_per_class()
    subset_total_elems = sum(v for k, v in single_labels)
    df = pd.DataFrame(single_labels, columns=['Class', 'Count'])
    df['Prevalence wrt. dataset'] = df['Count'].apply(lambda x: x / rcv1.target.shape[0])
    df['Prevalence wrt. S.L. subset'] = df['Count'].apply(lambda x: x / subset_total_elems)

    print(f"Dataframe for single-label rcv1: {df}")
    print("Saving to csv")
    df.to_csv('./rcv1_sl_hierarchy.tsv', sep='\t')


if __name__ == '__main__':
    save_csv_rcv1_single_label_hierarchy()
