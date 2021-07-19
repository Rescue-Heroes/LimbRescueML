from typing import OrderedDict

from sklearn.ensemble import RandomForestClassifier as MODEL


def get_default_hp_choices():
    hp_choices = OrderedDict(
        {
            "n_estimators": [10, 100, 1000],
            "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt", "log2"],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 3],
            "random_state": [0],
            "max_depth": [None, 10, 20, 30],
            "n_jobs": [-1],
            "ccp_alpha": [0.0, 0.2, 0.4, 0.6],
        }
    )
    return hp_choices


def get_default_hp_params():
    # train / val / test: 0.66 / 0.67 / 0.65
    best_hps = OrderedDict(
        {
            "n_estimators": 10,
            "criterion": "entropy",
            "max_features": "auto",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 0,
            "max_depth": None,
            "n_jobs": -1,
            "ccp_alpha": 0.2,
        }
    )
    return best_hps


if __name__ == "__main__":
    from pathlib import Path

    from utils import get_data, tune_datasets, tune_hyperparameters

    data_dir = Path(__file__).resolve().parent.parent
    data_dir = data_dir.joinpath("data")
    dataset = "ns10_ls300_normalized.npz"
    data = get_data(data_dir.joinpath(dataset))
    hp_choices = get_default_hp_choices()
    model, hp_params = tune_hyperparameters(MODEL, data, hp_choices)

    # hp_params = get_default_hp_params()
    datasets = sorted(list(data_dir.iterdir()))
    model, dataset = tune_datasets(MODEL, datasets, hp_params)
