import copy
import functools
import itertools
import logging
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from tqdm import tqdm

from limbresml.config.config import CfgNode as CN
from limbresml.config.config import get_cfg

logger = logging.getLogger(__name__)


def get_data(dataset_file):
    d = np.load(dataset_file)
    return {
        "X_train": d["X_train"],
        "y_train": d["y_train"],
        "X_val": d["X_val"],
        "y_val": d["y_val"],
        "X_test": d["X_test"],
        "y_test": d["y_test"],
    }
    # return d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]


def load_model(file):
    return load(file)["model"]


def save_model(model, file):
    model = {"model": model}
    dump(model, file)


def print_accuracy(acc_dict):
    s = " / ".join(acc_dict.keys()) + " accuracy"
    s += ": "
    s += " / ".join(["{:.2f}"] * len(acc_dict))
    logger.info(s.format(*acc_dict.values()))


def train_model(alg_module, cfg, data, print_acc=True):
    X_train, y_train, X_val, y_val, X_test, y_test = (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )
    cfg_model = cfg[cfg.MODEL]
    model = alg_module.get_model(cfg_model)
    model = model.fit(X_train, y_train)
    accuracy = OrderedDict(
        {
            "train": model.score(X_train, y_train),
            "val": model.score(X_val, y_val),
            "test": model.score(X_test, y_test),
        }
    )

    if print_acc:
        print_accuracy(accuracy)
    if cfg.OUTPUT_DIR:
        model_path = f"{cfg.OUTPUT_DIR}/{cfg.MODEL}.joblib"
        save_model(model, model_path)

    return model, accuracy


def tune_hyperparameters(alg_module, cfg, data):
    output_dir = cfg.OUTPUT_DIR
    cfg_model = cfg[cfg.MODEL]
    cfg_model_dict = {k: [v] for k, v in cfg_model.items()}

    hp_choices = alg_module.get_default_hp_choices()
    hp_choices.update(cfg_model_dict)
    hps = list(hp_choices.keys())
    choices = list(itertools.product(*list(hp_choices.values())))

    logger.info(
        f"Running {len(choices)} experiments with different combination of hyper-parameters..."
    )

    best_val = {"val": -1}
    _cfg = get_cfg(cfg.MODEL)
    for c in tqdm(choices):
        hp_params = dict(zip(hps, c))
        cfg_model = CN(hp_params)

        _cfg[cfg.MODEL].merge_from_other_cfg(cfg_model)
        _cfg.OUTPUT_DIR = ""

        clf, accuracy = train_model(alg_module, _cfg, data, False)
        if accuracy["val"] > best_val["val"]:
            best_val = copy.deepcopy(accuracy)
            best_model = clf
            best_hp_params = hp_params
            best_cfg = _cfg.clone()

    best_cfg.OUTPUT_DIR = output_dir
    if output_dir:
        model_path = f"{output_dir}/{cfg.MODEL}.joblib"
        save_model(best_model, model_path)
        cfg_path = f"{output_dir}/{cfg.MODEL}_best_hps.yaml"
        with open(cfg_path, "w") as f:
            f.write(best_cfg.dump())

    logger.info(f"Best validation accuracy {best_val['val']:.2f} by \n{str(best_hp_params)}")
    print_accuracy(best_val)

    return best_model, best_cfg


def tune_datasets(model, datasets, hp_params={}):
    logger.info(f"Running {len(datasets)} experiments with different preprocessing on data...")
    best_val = {"val": -1}
    for dataset in tqdm(datasets):
        data = get_data(dataset)
        clf, accuracy = train_model(model, data, hp_params, False)

        if accuracy["val"] > best_val["val"]:
            best_val = copy.deepcopy(accuracy)
            best_dataset = dataset
            best_model = clf

    logger.info(f"Best validation accuracy {best_val['val']:.2f} by {str(best_dataset)}")
    print_accuracy(best_val)
    return best_model, best_dataset


def gen_confusion_matrix(model, X, y, labels=None, plot_to=None):
    num_classes = len(model.classes_)
    if labels is None:
        labels = [f"Class{_+1}" for _ in range(num_classes)]
    pred_labels = ["Pred. " + _ for _ in labels]
    true_labels = ["True " + _ for _ in labels]

    cm = confusion_matrix(y, model.predict(X))
    print("Confustion Matrix")
    s = "{:<15}" * (len(model.classes_) + 1)
    print(s.format("", *pred_labels))
    for label, acc in zip(true_labels, cm):
        print(s.format(label, *acc))

    if plot_to is not None:
        titles_options = [" ", "normalized"]
        normal_options = [None, "true"]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        for title, normalize, ax in zip(titles_options, normal_options, axes.flatten()):
            plot_confusion_matrix(
                model, X, y, display_labels=labels, ax=ax, cmap=plt.cm.Blues, normalize=normalize
            )
            ax.title.set_text(title)

        plt.savefig(plot_to)


class _ColorfulFormatter(logging.Formatter):
    grey = "\x1b[38;1m"
    red_bg = "\x1b[41;1m"
    green_bg = "\x1b[46;1m"
    red = "\x1b[91;1m"
    green = "\x1b[92;1m"
    blue = "\x1b[94;1m"
    cyan = "\x1b[96;1m"
    reset = "\x1b[0m"

    fmt = "{}[%(asctime)s] %(name)s [%(levelname)s]:{} %(message)s{}"

    FORMATS = {
        logging.DEBUG: fmt.format(grey, "", reset),
        logging.INFO: fmt.format(green, reset, ""),
        logging.WARNING: fmt.format(red, "", reset),
        logging.ERROR: fmt.format(red_bg, reset, ""),
        logging.CRITICAL: fmt.format(cyan, reset, ""),
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(fmt, datefmt="%m/%d %H:%M:%S")
        return formatter.format(record)


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(name="limbresml"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = _ColorfulFormatter()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
