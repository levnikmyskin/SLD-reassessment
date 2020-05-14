import pickle
from datetime import date

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from dataset_helpers.utils import rcv1_binary_dataset
from dataset_helpers import Rcv1Helper
from em import em
from em_test import generate_n_randomly_prevalences_random_classes, generate_n_randomly_modified_prevalence
from load_data import get_measures_from_singlehist_measures

logging.basicConfig(filename="computation.log", level=logging.INFO, format='%(asctime)s:%(message)s')
device = torch.device('cuda')


class TrainDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return np.asarray(self.x[index].todense()).squeeze(), self.y[index]


class TestDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return np.asarray(self.x[index].todense()).squeeze()


class MLP(nn.Module):
    def __init__(self, n_input_features, n_output_classes, hidden_size=10000):
        super().__init__()
        self.layer_1 = nn.Linear(n_input_features, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, n_output_classes)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(self.layer_2(x))
        return F.softmax(x, dim=1)


def train_mlp(model, train_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print(f"Training MLP: {model}")
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch_num, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device).float()
            y = y.to(device).long()

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

        print(f"Epoch {epoch+1} | Loss {sum(losses) / len(losses):.4f}", end='\r', flush=True)


def predict(model, test_loader):
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device).float()
            yield model(x)


def em_on_mlp(y_tr, y_te, test_posteriors, n_classes):
    mlb = MultiLabelBinarizer(classes=range(n_classes))
    mlb.fit(np.expand_dims(np.hstack((y_tr, y_te)), 1))
    y_tr_bin = mlb.transform(np.expand_dims(y_tr, 1))
    train_priors = np.mean(y_tr_bin, 0)
    em_test_posteriors, em_test_priors, history = em(y_te, test_posteriors, train_priors, multi_class=n_classes > 2)
    measures = get_measures_from_singlehist_measures(history)

    return measures


def binary_experiments():
    rcv1_helper = Rcv1Helper()
    binary_dataset = rcv1_binary_dataset(rcv1_helper)
    n_classes = 2
    batch_size = 200

    for x, y, class_name in binary_dataset:
        logging.info(f"Running experiments for class {class_name}")
        gen = generate_n_randomly_modified_prevalence(500, x, y, 1000, 1000)
        measures = []
        count = 1
        for x_tr, y_tr, x_te, y_te in gen:
            train_set = TrainDataset(x_tr, y_tr)
            test_set = TestDataset(x_te)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_set, shuffle=False)

            model = MLP(x_tr.shape[1], n_classes, 5000).to(device)
            train_mlp(model, train_loader, epochs=80)
            probs = torch.cat(list(predict(model, test_loader))).cpu().numpy()
            measures.append(em_on_mlp(y_tr, y_te, probs, n_classes))
            print(f"Experiment {count} / 500 completed")
            count += 1

        logging.info(f"Saving measures for class {class_name}")
        with open(
                f'./pickles/measures_new_experiments/measures_500_rcv1{class_name}_{n_classes}_mlp_{date.today().strftime("%d-%m-%y")}.pkl',
                'wb') as f:
            pickle.dump(measures, f)


def single_label_experiments():
    pass


if __name__ == '__main__':
    binary_experiments()
