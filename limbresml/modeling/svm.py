from sklearn.svm import SVC

from ..config import CfgNode as CN
from ..config import cfg_value_to_list

"""
# -----------------------------------------------------------------------------
# Support Vector Machine
# For full documentation, please see:
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# -----------------------------------------------------------------------------
"""


def get_model(cfg_model):
    return SVC(
        C=cfg_model.C,
        kernel=cfg_model.kernel,
        degree=cfg_model.degree,
        gamma=cfg_model.gamma,
        coef0=cfg_model.coef0,
        shrinking=cfg_model.shrinking,
        probability=cfg_model.probability,
        tol=cfg_model.tol,
        cache_size=cfg_model.cache_size,
        class_weight=cfg_model.class_weight,
        verbose=cfg_model.verbose,
        max_iter=cfg_model.max_iter,
        decision_function_shape=cfg_model.decision_function_shape,
        break_ties=cfg_model.break_ties,
        random_state=cfg_model.random_state,
    )


def add_cfg_model(cfg, tune=False):
    cfg.SVM = CN()
    cfg.SVM.C = 7.0
    cfg.SVM.kernel = "rbf"  # {"linear", "poly", "rbf", "sigmoid", "precomputed"}
    cfg.SVM.degree = 1
    cfg.SVM.gamma = None  # {"scale", "auto"} or float, default="scale"
    cfg.SVM.coef0 = 0.0
    cfg.SVM.shrinking = True
    cfg.SVM.probability = False
    cfg.SVM.tol = 1e-3
    cfg.SVM.cache_size = 200.0
    cfg.SVM.class_weight = None  # dict or "balanced", default=None
    cfg.SVM.verbose = False
    cfg.SVM.max_iter = -1
    cfg.SVM.decision_function_shape = "ovo"  # {"ovo", "ovr"}
    cfg.SVM.break_ties = False
    cfg.SVM.random_state = None  # None or int

    if tune:
        cfg_value_to_list(cfg.SVM)
