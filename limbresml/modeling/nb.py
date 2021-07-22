from typing import OrderedDict

from sklearn.naive_bayes import GaussianNB


def get_model_class():
    return GaussianNB


def get_model(cfg_model):
    return GaussianNB(
        priors=cfg_model.priors,
        var_smoothing=cfg_model.var_smoothing,
    )


def get_default_hp_choices():
    # ---COMPLEMENT NAIVE BAYES---
    # model = ComplementNB
    # hp_choices = OrderedDict(
    # 	{
    # 		"alpha": [1],
    # 		"fit_prior": [True, False],
    # 		"class_prior": [None],
    # 		"norm": [True, False]
    # 	}
    # )

    # ---CATEGORICAL NAIVE BAYES---
    # model = CategoricalNB
    # hp_choices = OrderedDict(
    # 	{
    # 		# "alpha": [0,10],
    # 		"fit_prior": [True, False],
    # 		"class_prior": [None],
    # 		"min_categories": [None]
    # 	}
    # )

    # ---GAUSSIAN NAIVE BAYES---
    hp_choices = OrderedDict({"priors": [None], "var_smoothing": [1e-8, 1e-10, 1e-11, 1e-9, 1e-12]})
    return hp_choices


def get_default_hp_params():
    # train / val / test: 0.39 / 0.58 / 0.33
    best_hps = OrderedDict(
        {
            "priors": None,
            "var_smoothing": 1e-08,
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
