from load_data import Measures
import matplotlib.pyplot as plt
import numpy as np
import pickle


if __name__ == '__main__':
    with open('./pickles/measures_all_em.pkl', 'rb') as f:
        data = iter(pickle.load(f))

    while input("Enter to continue, any other key+enter to stop") == "":
        clf_name, measures = next(data)
        print(f"Processing data for classifier {clf_name}")
        max_len = max(len(elem.em_abs_errors) for elem in measures)
        padded_abs_errors = np.array([np.pad(elem.abs_errors, (0, max_len - len(elem.abs_errors)), mode='edge') for elem in measures])
        padded_em_abs_errors = np.array([np.pad(elem.em_abs_errors, (0, max_len - len(elem.em_abs_errors)), mode='edge') for elem in measures])

        mean_abs_errors = np.vstack(padded_abs_errors).mean(axis=0)
        mean_em_abs_errors = np.vstack(padded_em_abs_errors).mean(axis=0)

        p1 = plt.plot(range(len(mean_abs_errors)), mean_em_abs_errors)
        p2 = plt.plot(range(len(mean_abs_errors)), np.vstack(padded_em_abs_errors).var(axis=0))

        plt.legend((p1[0], p2[0]), ('Mean', 'Variance'))
        plt.show()

