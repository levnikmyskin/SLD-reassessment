from latex_results import load_measure_for_classifiers_pickles
from load_data import Measures
import numpy as np
import pickle
import itertools


def normalize_ref_cal(measure: Measures, labels_len: int):
    isomer_ref_loss = np.array(measure.isomerous_em_ref_loss[1])
    isomer_cal_loss = np.array(measure.isomerous_em_cal_loss[1])
    isomet_ref_loss = np.array(measure.isometric_em_ref_loss[1])
    isomet_cal_loss = np.array(measure.isometric_em_cal_loss[1])

    norm_factor = labels_len * 10  # 10 is the bin numbers

    return isomer_ref_loss / norm_factor, isomer_cal_loss / norm_factor, \
           isomet_ref_loss / norm_factor, isomet_cal_loss / norm_factor


def compress_list(data):
    args = [iter(data)] * 100
    return list(itertools.zip_longest(*args))


if __name__ == '__main__':
    class_num = '2'
    dataset = 'rcv1'
    measures_gen = load_measure_for_classifiers_pickles(dataset, class_num, with_file_name=True)
    clf, data, file_name = next(measures_gen)

    for clf, data, file_name in measures_gen:
        for i, measure in enumerate(data):
            print(f"Normalizing brier on {file_name}: {i}/{len(data)}", end='\r', flush=True)
            isomer_ref_loss, isomer_cal_loss, isomet_ref_loss, isomet_cal_loss = normalize_ref_cal(measure, int(class_num))
            assert (isomer_ref_loss + isomer_cal_loss).any() <= 1 and (isomet_ref_loss + isomet_cal_loss).any() \
                   <= 1, f"Brier is greater than 1 for {file_name}"
            data[i] = measure._replace(
                isomerous_em_ref_loss=(measure.isomerous_em_ref_loss[0], isomer_ref_loss),
                isomerous_em_cal_loss=(measure.isomerous_em_cal_loss[0], isomer_cal_loss),
                isometric_em_ref_loss=(measure.isometric_em_ref_loss[0], isomet_ref_loss),
                isometric_em_cal_loss=(measure.isometric_em_cal_loss[0], isomet_cal_loss)
            )
        with open(file_name, 'wb') as f:
            pickle.dump(compress_list(data), f)
