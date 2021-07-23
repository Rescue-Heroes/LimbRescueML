import argparse
import itertools
import logging
from pathlib import Path

from limbresml.config.config import get_cfg
from limbresml.utils import (
    generate_confusion_matrix,
    get_data,
    setup_logger,
    train_model,
    tune_datasets,
    tune_hyperparameters,
)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    setup_logger("limbresml")
    # default_setup(cfg, args)
    return cfg


# def get_hp(algorithm, tune, customized_hp):
#     if tune:
#         hp_choices = algorithm.get_default_hp_choices()
#         model, hp_params = tune_hyperparameters(algorithm.MODEL, data, hp_choices)
#         print(hp_params)
#     hp_params = algorithm.get_default_hp_params()
#     if customized_hp:
#         customized_hp = ["".join(_) for _ in customized_hp]

#         customized_hp = iter(customized_hp)
#         customized_hp_dic = dict([[_, next(customized_hp)] for _ in customized_hp])
#         print(customized_hp_dic)
#     return


def parse_args():
    parser = argparse.ArgumentParser(description="Train machine learning model. ")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        default="configs/svm.yaml",
        type=Path,
        help="the config file",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="perform evaluation only",
    )
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # parser.add_argument(
    #     "--dataset-file",
    #     metavar="PATH",
    #     default="data/ns10_ls300_normalized.npz",
    #     type=Path,
    #     help="the preprocessed data files",
    # )
    # parser.add_argument(
    #     "--output-dir",
    #     metavar="DIR",
    #     default="experiment/",
    #     type=Path,
    #     help="the directory for outputs",
    # )
    # parser.add_argument(
    #     "--algorithm",
    #     metavar="ALGORITHM",
    #     default="svm",
    #     type=str,
    #     choices=["svm", "mlp", "random_forest", "gaussian_nb"],
    #     help="machine learning algorithm, choices including: \
    #         svm(Support Vector Machine), mlp(Multilayer Perceptron),\
    #         random_forest(Random Forest) and gaussian_naive_bayes(Gaussian Naive Bayes)",
    # )
    # parser.add_argument(
    #     "--tune",
    #     metavar="TRUE",
    #     default=False,
    #     type=bool,
    #     help="tune hyerparameters or using default,\
    #         choices including: True(tune) and False(default)",
    # )
    # parser.add_argument(
    #     "--save-confusion-matrix",
    #     "-save-cm",
    #     metavar="TRUE",
    #     default=False,
    #     type=bool,
    #     help="save confusion matrix or not,\
    #         choices including: True and False(default)",
    # )
    # parser.add_argument(
    #     "--combinedata",
    #     metavar="TRUE",
    #     default=False,
    #     type=bool,
    #     help="train model with combined train and validation data or train data only,\
    #         choices including: True and False(default)",
    # )
    # parser.add_argument(
    #     "--hyper-pars",
    #     metavar="LIST",
    #     default=None,
    #     type=list,
    #     nargs="+",
    #     help="customized hyperparameters for training model,\
    #         formatted as key value key value...",
    # )
    return parser.parse_args()


if __name__ == "__main__":
    # import importlib

    from limbresml.modeling.svm import get_model

    args = parse_args()
    cfg = setup(args)
    hp = cfg[cfg["MODEL"]]
    get_model(hp)
    # data = get_data(args.dataset_file)
    # algorithm = importlib.import_module(f"limbresml.{args.algorithm}")
    # hp = get_hp(algorithm, args.tune, args.hyper_pars)
