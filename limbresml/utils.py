import numpy as np
from joblib import dump, load

def get_data(dataset_file):
    d = np.load(dataset_file)
    return d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]

def load_model(file):
    return load(file)["model"]

def save_model(model, file, best_hps=None, dataset=None):
    model = {"model": model, "hps": best_hps, "dataset": dataset}
    dump(model, file)    