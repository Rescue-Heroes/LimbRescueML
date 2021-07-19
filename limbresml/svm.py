from typing import OrderedDict

import numpy as np
from sklearn.svm import SVC as MODEL


def get_default_hp_choices():
    hp_choices = OrderedDict(
        {
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "degree": list(range(1, 6)),
            "gamma": ["scale", "auto"],
            "C": np.linspace(1, 10, num=10).tolist(),
            "decision_function_shape": ["ovo", "ovr"],
            "class_weight": ["balanced", None],
        }
    )
    return hp_choices


def get_default_hp_params():
    # train / val / test: 0.82 / 0.75 / 0.67
    best_hps = OrderedDict(
        {
            "kernel": "rbf",
            "degree": 1,
            "gamma": "scale",
            "C": 7.0,
            "decision_function_shape": "ovo",
            "class_weight": None,
        }
    )
    return best_hps


if __name__ == "__main__":
    from pathlib import Path

    from utils import (
        generate_confusion_matrix,
        get_data,
        train_model,
        tune_datasets,
        tune_hyperparameters,
    )

    data_dir = Path(__file__).resolve().parent.parent
    data_dir = data_dir.joinpath("data")
    dataset = "ns10_ls300_normalized.npz"
    data = get_data(data_dir.joinpath(dataset))
    hp_choices = get_default_hp_choices()
    # model, hp_params = tune_hyperparameters(MODEL, data, hp_choices)

    hp_params = get_default_hp_params()
    model, _ = train_model(MODEL, data, hp_params, print_acc=True)
    generate_confusion_matrix(model, data["X_val"], data["y_val"], plot=True)
    # datasets = sorted(list(data_dir.iterdir()))
    # model, dataset = tune_datasets(MODEL, datasets, hp_params)
