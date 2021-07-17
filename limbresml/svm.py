#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from typing import OrderedDict
from pathlib import Path

import numpy as np
from sklearn import svm
from tqdm import tqdm
from joblib import dump, load

from utils import get_data
from utils import save_model

model_name = "svm2.joblib"
data_dir = Path(__file__).resolve().parent.parent
data_dir = data_dir.joinpath("data2")
dataset = "ns10_ls300_normalized.npz"
X_train, y_train, X_val, y_val, X_test, y_test = get_data(data_dir.joinpath(dataset))

# for y in [y_train, y_val, y_test]:
# 	y[y==2] = 4
# 	y[y==3] = 2
# 	y[y==4] = 3


model = svm.SVC
hyperparameters = OrderedDict(
	{
		"kernel": ["linear", "rbf", "poly", "sigmoid"],
		"degree": list(range(1, 6)),
		"gamma": ["scale", "auto"],
		"C": np.linspace(1, 10, num=10).tolist(),
		"decision_function_shape": ["ovo", "ovr"],
		"class_weight": ["balanced", None]
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
	clf = model(**kwargs).fit(X_train, y_train)
	_acc_train = clf.score(X_train, y_train)
	_acc_val = clf.score(X_val, y_val)
	_acc_test = clf.score(X_test, y_test)
	# print(f"{str(kwargs)}\n train / val / test: {_acc_train:.2f} / {_acc_val:.2f} / {_acc_test:.2f}\n")
	if _acc_val > best_val:
		best_val = _acc_val
		best_hps = kwargs

		acc_train = _acc_train
		acc_test = _acc_test
		
print(f"Best validation accuracy {best_val:.2f} by {str(best_hps)}")
print(f"train / val / test: {acc_train:.2f} / {best_val:.2f} / {acc_test:.2f}\n")

save_model(model(**best_hps).fit(X_train, y_train), model_name, best_hps, dataset)

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
	# print(f"{str(kwargs)}\n train / val / test: {_acc_train:.2f} / {_acc_val:.2f} / {_acc_test:.2f}\n")
	if _acc_val > best_val:
		best_val = _acc_val
		best_dataset = dataset

		acc_train = _acc_train
		acc_test = _acc_test

print(f"Best validation accuracy {best_val:.2f} by {str(best_dataset)}")
print(f"train / val / test: {acc_train:.2f} / {best_val:.2f} / {acc_test:.2f}")
