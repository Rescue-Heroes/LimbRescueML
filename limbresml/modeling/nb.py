from sklearn.naive_bayes import GaussianNB

from ..config import CfgNode as CN
from ..config import cfg_value_to_list

"""
# -----------------------------------------------------------------------------
# Gaussian Naive Bayes
# For full documentation, please see:
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
# -----------------------------------------------------------------------------
"""


def get_model(cfg_model):
    return GaussianNB(
        priors=cfg_model.priors,
        var_smoothing=cfg_model.var_smoothing,
    )


def add_cfg_model(cfg, tune=False):
    cfg.NB = CN()
    cfg.NB.priors = None  # array-like of shape(n_classes,)
    cfg.NB.var_smoothing = 1e-08

    if tune:
        cfg_value_to_list(cfg.NB)
