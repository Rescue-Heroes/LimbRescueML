from typing import OrderedDict

from sklearn.ensemble import RandomForestClassifier

from ..config import CfgNode as CN
from ..config import cfg_value_to_list

"""
# -----------------------------------------------------------------------------
# Random Forest
# For full documentation, please see:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# -----------------------------------------------------------------------------
"""


def get_model(cfg_model):
    return RandomForestClassifier(
        n_estimators=cfg_model.n_estimators,
        criterion=cfg_model.criterion,
        max_depth=cfg_model.max_depth,
        min_samples_split=cfg_model.min_samples_split,
        min_samples_leaf=cfg_model.min_samples_leaf,
        min_weight_fraction_leaf=cfg_model.min_weight_fraction_leaf,
        max_features=cfg_model.max_features,
        max_leaf_nodes=cfg_model.max_leaf_nodes,
        min_impurity_decrease=cfg_model.min_impurity_decrease,
        min_impurity_split=cfg_model.min_impurity_split,
        bootstrap=cfg_model.bootstrap,
        oob_score=cfg_model.oob_score,
        n_jobs=cfg_model.n_jobs,
        random_state=cfg_model.random_state,
        verbose=cfg_model.verbose,
        warm_start=cfg_model.warm_start,
        class_weight=cfg_model.class_weight,
        ccp_alpha=cfg_model.ccp_alpha,
        max_samples=cfg_model.max_samples,
    )


def add_cfg_model(cfg, tune=False):
    cfg.RF = CN()
    cfg.RF.n_estimators = 10
    cfg.RF.criterion = "entropy"  # {"gini", "entropy"}
    cfg.RF.max_depth = None  # None or int
    cfg.RF.min_samples_split = None  # int or float, default=2
    cfg.RF.min_samples_leaf = None  # int or float, default=1
    cfg.RF.min_weight_fraction_leaf = 0.0
    cfg.RF.max_features = None  # {"auto", "sqrt", "log2"}, int or float, default="auto"
    cfg.RF.max_leaf_nodes = None  # None or int
    cfg.RF.min_impurity_decrease = 0.0
    cfg.RF.min_impurity_split = None  # None or float
    cfg.RF.bootstrap = True
    cfg.RF.oob_score = False
    cfg.RF.n_jobs = None  # None or int
    cfg.RF.random_state = None  # None or int
    cfg.RF.verbose = 0
    cfg.RF.warm_start = False
    cfg.RF.class_weight = None  # {None, "balanced", "balanced_subsample"}, dict or list of dicts
    cfg.RF.ccp_alpha = 0.2
    cfg.RF.max_samples = None  # None, int or float

    if tune:
        cfg_value_to_list(cfg.RF)
