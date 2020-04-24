import pickle
import numpy as np
from sklearn.datasets import fetch_rcv1
from sklearn.metrics import brier_score_loss
from em import soft_accuracy, soft_f1


with open('/home/alessio/minecore/pickles/emq_posteriors_full.pkl', 'rb') as f:
    emq_post = pickle.load(f)

with open('/home/alessio/minecore/pickles/full_posteriors.pkl', 'rb') as f:
    pre_post = pickle.load(f)

with open('/home/alessio/minecore/pickles/bef_emq_priors.pkl', 'rb') as f:
    pre_priors = pickle.load(f)

with open('/home/alessio/minecore/pickles/emq_priors.pkl', 'rb') as f:
    emq_priors = pickle.load(f)


TRAINING_SET_END = 23149
SMALL_TEST_SET_END = TRAINING_SET_END + 199328
TEST_SET_START = TRAINING_SET_END
TEST_SET_END = SMALL_TEST_SET_END

dataset = fetch_rcv1()
full_test_set_end = dataset.target.shape[0]
quarter_y_arr = dict()
full_y_arr = dict()
training_y_arr = dict()

for i, c in enumerate(dataset.target_names):
    quarter_y_arr[c] = np.asarray(dataset.target[TEST_SET_START:TEST_SET_END, i].todense()).squeeze()
    full_y_arr[c] = np.asarray(dataset.target[TEST_SET_START:, i].todense()).squeeze()
    training_y_arr[c] = np.asarray(dataset.target[0:TRAINING_SET_END, i].todense()).squeeze()

cls = 'GCAT'
emq_cls_post = emq_post[cls]
pre_cls_post = pre_post[cls]
y = full_y_arr[cls]

print(f"EM Brier: {brier_score_loss(y, emq_cls_post[:, 1])}")
print(f"MLE Brier: {brier_score_loss(y, pre_cls_post[:, 1])}")
print(f"MLE abs: {abs(y.mean() - pre_priors[cls][1])}")
print(f"EMQ abs: {abs(y.mean() - emq_priors[cls][1])}")

print(f"EMQ soft acc: {soft_accuracy(y, emq_post[cls])}")
print(f"MLE soft acc: {soft_accuracy(y, pre_post[cls])}")

print(f"EMQ soft F1: {soft_f1(y, emq_post[cls])}")
print(f"MLE soft F1: {soft_f1(y, pre_post[cls])}")
