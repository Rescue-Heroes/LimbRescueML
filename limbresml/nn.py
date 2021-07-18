from sklearn.neural_network import MLPClassifier as MODEL
from typing import OrderedDict


def get_default_hp_choices():
    hp_choices = OrderedDict(
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
    return hp_choices


def get_default_hp_params():
    # train / val / test: 0.90 / 0.80 / 0.68
    best_hps = OrderedDict(
        {
            "hidden_layer_sizes": (300,),
            "activation": "logistic",
            "solver": "adam",
            "learning_rate": "adaptive",
            "max_iter": 10000000000,
            "random_state": 0,
            "n_iter_no_change": 1000,
        }
    )
    return best_hps


if __name__ == "__main__":
    from pathlib import Path
    from utils import get_data, tune_hyperparameters, tune_datasets

    data_dir = Path(__file__).resolve().parent.parent
    data_dir = data_dir.joinpath("data")
    dataset = "ns10_ls300_normalized.npz"
    data = get_data(data_dir.joinpath(dataset))
    hp_choices = get_default_hp_choices()
    model, hp_params = tune_hyperparameters(MODEL, data, hp_choices)

    # hp_params = get_default_hp_params()
    datasets = sorted(list(data_dir.iterdir()))
    model, dataset = tune_datasets(MODEL, datasets, hp_params)
