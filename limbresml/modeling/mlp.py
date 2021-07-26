from typing import OrderedDict

from sklearn.neural_network import MLPClassifier


def get_model_class():
    return MLPClassifier


def get_model(cfg_model):
    return MLPClassifier(
        hidden_layer_sizes=cfg_model.hidden_layer_sizes,
        activation=cfg_model.activation,
        solver=cfg_model.solver,
        alpha=cfg_model.alpha,
        batch_size=cfg_model.batch_size,
        learning_rate=cfg_model.learning_rate,
        learning_rate_init=cfg_model.learning_rate_init,
        power_t=cfg_model.power_t,
        max_iter=cfg_model.max_iter,
        shuffle=cfg_model.shuffle,
        random_state=cfg_model.random_state,
        tol=cfg_model.tol,
        verbose=cfg_model.verbose,
        warm_start=cfg_model.warm_start,
        momentum=cfg_model.momentum,
        nesterovs_momentum=cfg_model.nesterovs_momentum,
        early_stopping=cfg_model.early_stopping,
        validation_fraction=cfg_model.validation_fraction,
        beta_1=cfg_model.beta_1,
        beta_2=cfg_model.beta_2,
        epsilon=cfg_model.epsilon,
        n_iter_no_change=cfg_model.n_iter_no_change,
        max_fun=cfg_model.max_fun,
    )


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
            # "random_state": [None],
            "n_iter_no_change": [1000],
            "batch_size": "auto",
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
