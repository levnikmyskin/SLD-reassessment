from latex_results import load_measure_for_classifiers_pickles
from load_data import get_measures_mean_across_experiments, Measures


gen = load_measure_for_classifiers_pickles("rcv1")
data = next(gen)[1]

mm = get_measures_mean_across_experiments(data)

