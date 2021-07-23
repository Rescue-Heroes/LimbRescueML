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


def save_model(model, file, best_hps=None, dataset=None):
    model = {"model": model, "hps": best_hps, "dataset": dataset}
    dump(model, file)


def print_accuracy(acc_dict):
    s = " / ".join(acc_dict.keys())
    s += ": "
    s += " / ".join(["{:.2f}"] * len(acc_dict))
    print(s.format(*acc_dict.values()))


def train_model(model, data, hp_params={}, print_acc=True):
    X_train, y_train, X_val, y_val, X_test, y_test = (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )
    model = model(**hp_params).fit(X_train, y_train)
    accuracy = OrderedDict(
        {
            "train": model.score(X_train, y_train),
            "val": model.score(X_val, y_val),
            "test": model.score(X_test, y_test),
        }
    )
    if print_acc:
        print_accuracy(accuracy)

    return model, accuracy


def train_model(cfg, data, print_acc=True):
    import importlib

    model = importlib.import_module(f"modeling.{cfg.MODEL}")
    model
    X_train, y_train, X_val, y_val, X_test, y_test = (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )
    model = model(**hp_params).fit(X_train, y_train)
    accuracy = OrderedDict(
        {
            "train": model.score(X_train, y_train),
            "val": model.score(X_val, y_val),
            "test": model.score(X_test, y_test),
        }
    )
    if print_acc:
        print_accuracy(accuracy)

    return model, accuracy


def tune_hyperparameters(model, data, hp_choices):
    hps = list(hp_choices.keys())
    choices = list(itertools.product(*list(hp_choices.values())))
    print(f"Running {len(choices)} experiments with different combination of hyper-parameters...")

    best_val = {"val": -1}
    for c in tqdm(choices):
        hp_params = dict(zip(hps, c))
        clf, accuracy = train_model(model, data, hp_params, False)

        if accuracy["val"] > best_val["val"]:
            best_val = copy.deepcopy(accuracy)
            best_hp_params = hp_params
            best_model = clf

    print(f"Best validation accuracy {best_val['val']:.2f} by {str(best_hp_params)}")
    print_accuracy(best_val)
    return best_model, best_hp_params


def tune_datasets(model, datasets, hp_params={}):
    print(f"Running {len(datasets)} experiments with different preprocessing on data...")
    best_val = {"val": -1}
    for dataset in tqdm(datasets):
        data = get_data(dataset)
        clf, accuracy = train_model(model, data, hp_params, False)

        if accuracy["val"] > best_val["val"]:
            best_val = copy.deepcopy(accuracy)
            best_dataset = dataset
            best_model = clf

    print(f"Best validation accuracy {best_val['val']:.2f} by {str(best_dataset)}")
    print_accuracy(best_val)
    return best_model, best_dataset


def generate_confusion_matrix(model, X, y, labels=None, plot=False, file=None):
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

    if plot is True:
        titles_options = [" ", "normalized"]
        normal_options = [None, "true"]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        for title, normalize, ax in zip(titles_options, normal_options, axes.flatten()):
            plot_confusion_matrix(
                model, X, y, display_labels=labels, ax=ax, cmap=plt.cm.Blues, normalize=normalize
            )
            ax.title.set_text(title)
        if file:
            plt.savefig(file)
        plt.show()


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
