#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from typing import OrderedDict
from pathlib import Path

import numpy as np
from sklearn import ensemble
from tqdm import tqdm

from utils import get_data
from utils import save_model

BEST_HPS = {'n_estimators': 10, 'criterion': 'gini', 'max_features': 'auto',
            'min_samples_split': 3, 'min_samples_leaf': 1, 'random_state': None}
best_hps = BEST_HPS

model_name = "rf.joblib"
data_dir = Path(__file__).resolve().parent.parent
data_dir = data_dir.joinpath("data")
dataset = "ns20_ls100_first_order.npz"
X_train, y_train, X_val, y_val, X_test, y_test = get_data(
    data_dir.joinpath(dataset))


model = ensemble.RandomForestClassifier
# hyperparameters = OrderedDict(
#     {
#         "n_estimators": [10, 100, 1000],
#         "criterion": ["gini", "entropy"],
#         "max_features": ["auto", "sqrt", "log2"],
#         "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "min_samples_leaf": [1, 2, 3],
#         "random_state": [0, None]
#     }
# )

# hps = list(hyperparameters.keys())
# choices = list(itertools.product(*list(hyperparameters.values())))
# print(
#     f"Running {len(choices)} experiments with different combinatoin of hyper-parameters...")

# acc_train = 0
# best_val = 0
# acc_test = 0
# best_hps = None
# for c in tqdm(choices):
#     kwargs = dict(zip(hps, c))
#     clf = model(**kwargs).fit(X_train, y_train)
#     _acc_train = clf.score(X_train, y_train)
#     _acc_val = clf.score(X_val, y_val)
#     _acc_test = clf.score(X_test, y_test)
#     # print(f"{str(kwargs)}\n train / val / test: {_acc_train:.2f} / {_acc_val:.2f} / {_acc_test:.2f}\n")
#     if _acc_val > best_val:
#         best_val = _acc_val
#         best_hps = kwargs

#         acc_train = _acc_train
#         acc_test = _acc_test

# print(f"Best validation accuracy {best_val:.2f} by {str(best_hps)}")
# print(
#     f"train / val / test: {acc_train:.2f} / {best_val:.2f} / {acc_test:.2f}\n")

save_model(model(**best_hps).fit(X_train, y_train),
           model_name, best_hps, dataset)


# datasets = sorted(list(data_dir.iterdir()))
# print(
#     f"Running {len(datasets)} experiments with different preprocessing on data...")
# acc_train = 0
# best_val = 0
# acc_test = 0
# best_dataset = None
# for dataset in tqdm(datasets):
#     X_train, y_train, X_val, y_val, X_test, y_test = get_data(
#         data_dir.joinpath(dataset))
#     clf = model(**best_hps).fit(X_train, y_train)
#     _acc_train = clf.score(X_train, y_train)
#     _acc_val = clf.score(X_val, y_val)
#     _acc_test = clf.score(X_test, y_test)
#     # print(f"{str(kwargs)}\n train / val / test: {_acc_train:.2f} / {_acc_val:.2f} / {_acc_test:.2f}\n")
#     if _acc_val > best_val:
#         best_val = _acc_val
#         best_dataset = dataset

#         acc_train = _acc_train
#         acc_test = _acc_test

# print(f"Best validation accuracy {best_val:.2f} by {str(best_dataset)}")
# print(f"train / val / test: {acc_train:.2f} / {best_val:.2f} / {acc_test:.2f}")
