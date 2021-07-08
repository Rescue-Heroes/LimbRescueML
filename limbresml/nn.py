#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from typing import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from utils import get_data

BEST_HPS = {
    "hidden_layer_sizes": (300,),
    "activation": "logistic",
    "solver": "adam",
    "learning_rate": "adaptive",
    "max_iter": 10000000000,
    "random_state": 0,
    "n_iter_no_change": 1000,
}

data_dir = Path(__file__).resolve().parent.parent
data_dir = data_dir.joinpath("data")
X_train, y_train, X_val, y_val, X_test, y_test = get_data(
    data_dir.joinpath("ns10_ls300_normalized.npz")
)

model = MLPClassifier
hyperparameters = OrderedDict(
    {
        "hidden_layer_sizes": [(300,), (300, 300), (300, 150)],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        # "alpha": [0.1, 0.01, 0.001, 0.0001, 0.00001],  # l2 penalty
        "learning_rate": ["adaptive"],
        # "learning_rate_init": [0.001],
        "max_iter": [int(1e10)],
        "random_state": [0],
        "n_iter_no_change": [1000],
    }
)
hps = list(hyperparameters.keys())
choices = list(itertools.product(*list(hyperparameters.values())))
print(f"Running {len(choices)} experiments with different combination of hyper-parameters...")

acc_train = 0
best_val = 0
acc_test = 0
best_hps = None
for c in tqdm(choices):
    kwargs = dict(zip(hps, c))
    clf = model(**kwargs)
    print(clf)
    clf = clf.fit(X_train, y_train)
    _acc_train = clf.score(X_train, y_train)
    _acc_val = clf.score(X_val, y_val)
    _acc_test = clf.score(X_test, y_test)
    print(
        f"{str(kwargs)}\n train / val / test: {_acc_train:.2f} / {_acc_val:.2f} / {_acc_test:.2f}\n"
    )
    if _acc_val > best_val:
        best_val = _acc_val
        best_hps = kwargs

        acc_train = _acc_train
        acc_test = _acc_test

print(f"Best validation accuracy {best_val:.2f} by {str(best_hps)}")
print(f"train / val / test: {acc_train:.2f} / {best_val:.2f} / {acc_test:.2f}\n")

datasets = sorted(list(data_dir.iterdir()))
print(f"Running {len(datasets)} experiments with different preprocessing on data...")
acc_train = 0
best_val = 0
acc_test = 0
best_dataset = None
for dataset in tqdm(datasets):
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(data_dir.joinpath(dataset))
    clf = model(**best_hps).fit(X_train, y_train)
    _acc_train = clf.score(X_train, y_train)
    _acc_val = clf.score(X_val, y_val)
    _acc_test = clf.score(X_test, y_test)
    print(
        f"{str(best_hps)}\n train / val / test: {_acc_train:.2f} / {_acc_val:.2f} / {_acc_test:.2f}\n"
    )
    if _acc_val > best_val:
        best_val = _acc_val
        best_dataset = dataset

        acc_train = _acc_train
        acc_test = _acc_test

print(f"Best validation accuracy {best_val:.2f} by {str(best_dataset)}")
print(f"train / val / test: {acc_train:.2f} / {best_val:.2f} / {acc_test:.2f}")
