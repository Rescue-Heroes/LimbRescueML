import numpy as np


def get_data(dataset_file):
    d = np.load(dataset_file)
    return d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]
