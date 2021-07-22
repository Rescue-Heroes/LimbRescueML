from typing import OrderedDict

import numpy as np
from sklearn.svm import SVC


def get_model_class():
    return SVC


def get_model(cfg_model):
    return SVC(
        C=cfg_model.C,
        kernel=cfg_model.kernel,
        degree=cfg_model.degree,
        gamma=cfg_model.gamma,
        coef0=cfg_model.coef0,
        shrinking=cfg_model.shrinking,
        probability=cfg_model.probability,
        tol=cfg_model.tol,
        cache_size=cfg_model.cache_size,
        class_weight=cfg_model.class_weight,
        verbose=cfg_model.verbose,
        max_iter=cfg_model.max_iter,
        decision_function_shape=cfg_model.decision_function_shape,
        break_ties=cfg_model.break_ties,
        random_state=cfg_model.random_state,
    )


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


def test():
    import logging

    logger = logging.getLogger(__name__)
    logger.info("test")
    print("test1")


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
    # dataset = "ns10_ls300_normalized.npz"
    dataset = "dataset.npz"
    data = get_data(data_dir.joinpath(dataset))
    hp_choices = get_default_hp_choices()
    model, hp_params = tune_hyperparameters(MODEL, data, hp_choices)

    hp_params = get_default_hp_params()
    model, _ = train_model(MODEL, data, hp_params, print_acc=True)
    generate_confusion_matrix(model, data["X_val"], data["y_val"], plot=True)
    # datasets = sorted(list(data_dir.iterdir()))
    # model, dataset = tune_datasets(MODEL, datasets, hp_params)
