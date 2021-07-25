import argparse
import logging
import os

from limbresml.config.config import CfgNode as CN
from limbresml.config.config import get_cfg
from limbresml.utils import (
    gen_confusion_matrix,
    get_data,
    setup_logger,
    train_model,
    tune_hyperparameters,
)

logger = logging.getLogger(__name__)
# from pathlib import Path


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = CN(CN.load_yaml(args.config_file))
    if not cfg.TUNE_HP_PARAMS:
        _cfg = get_cfg(cfg.MODEL)
        _cfg.merge_from_other_cfg(cfg)
        cfg = _cfg
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cfg_path = f"{output_dir}/config_backup.yaml"

        with open(cfg_path, "w") as f:
            f.write(cfg.dump())
    setup_logger("limbresml")
    return cfg


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
    import importlib

    args = parse_args()
    cfg = setup(args)

    dataset = cfg.INPUT.PATH
    data = get_data(dataset)

    X_train, y_train, X_val, y_val, X_test, y_test = (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )

    algorithm = cfg.MODEL
    # cfg_model = cfg[algorithm]
    alg_module = importlib.import_module(f"limbresml.modeling.{algorithm.lower()}")

    if cfg.TUNE_HP_PARAMS:
        model, cfg = tune_hyperparameters(alg_module, cfg, data)
    else:
        model, accuracy = train_model(alg_module, cfg, data)

    plot_to = f"{cfg.OUTPUT_DIR}/confusion_matrix.png"
    gen_confusion_matrix(model, X_val, y_val, plot_to=plot_to, labels=["normal", "left", "right"])
