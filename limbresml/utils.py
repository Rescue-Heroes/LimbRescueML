import copy
import itertools
from typing import OrderedDict
from joblib import dump, load
import numpy as np
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