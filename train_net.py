import argparse
import importlib
import logging
import os

from limbresml.config.config import CfgNode as CN
from limbresml.config.config import get_cfg
from limbresml.utils import (
    get_data,
    plot_confusion_matrix,
    setup_logger,
    train_model,
    tune_hyperparameters,
)

logger = logging.getLogger(__name__)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    logger = setup_logger("limbresml")
    cfg_file = CN(CN.load_yaml(args.config_file))
    model_name = cfg_file.get("MODEL", "svm")
    tune_hp_params = cfg_file.get("TUNE_HP_PARAMS", False)
    model_module = importlib.import_module(f"limbresml.modeling.{model_name.lower()}")

    cfg = get_cfg()
    model_module.add_cfg_model(cfg, tune_hp_params)
    cfg.merge_from_other_cfg(cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    logger.info(f"config: \n{str(cfg)}")

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cfg_path = f"{output_dir}/config_backup.yaml"

        logger.info(f"backup config file to {cfg_path}")
        with open(cfg_path, "w") as f:
            f.write(cfg.dump())

    return cfg, model_module


def parse_args():
    parser = argparse.ArgumentParser(description="Train machine learning model. ")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        default="configs/svm.yaml",
        type=str,
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
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    cfg, model_module = setup(args)
    logger = logging.getLogger("limbresml.train_net")

    dataset = cfg.INPUT.PATH
    data = get_data(dataset, cfg.INPUT.CAT_TRAIN_VAL)

    if cfg.TUNE_HP_PARAMS:
        logger.info(f"tune hyper-parameters of {cfg.MODEL} model...")
        model, accuracy = tune_hyperparameters(
            model_module, cfg, data
        )  # cfg will be modified in place by best cfg model
        print(cfg)
    else:
        logger.info(f"train {cfg.MODEL} model...")
        model, accuracy = train_model(model_module, cfg, data)

    for set in ["val", "test"]:
        X = data.get("X_" + set, None)
        if X is None:
            continue

        y = data["y_" + set]
        plot_to = f"{cfg.OUTPUT_DIR}/confusion_matrix_{set}.png"
        confusion_str = plot_confusion_matrix(
            model, X, y, plot_to=plot_to, labels=["normal", "left", "right"]
        )
        logger.info(f"confusion matrix for {set} set: \n{confusion_str}")
