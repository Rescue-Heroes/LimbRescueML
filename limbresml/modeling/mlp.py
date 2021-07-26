from sklearn.neural_network import MLPClassifier

from ..config import CfgNode as CN
from ..config import cfg_value_to_list

"""
# -----------------------------------------------------------------------------
# Multilayer Perceptron
# For full documentation, please see:
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# -----------------------------------------------------------------------------
"""


def get_model(cfg_model):
    return MLPClassifier(
        hidden_layer_sizes=cfg_model.hidden_layer_sizes,
        activation=cfg_model.activation,
        solver=cfg_model.solver,
        alpha=cfg_model.alpha,
        batch_size=cfg_model.batch_size,
        learning_rate=cfg_model.learning_rate,
        learning_rate_init=cfg_model.learning_rate_init,
        power_t=cfg_model.power_t,
        max_iter=cfg_model.max_iter,
        shuffle=cfg_model.shuffle,
        random_state=cfg_model.random_state,
        tol=cfg_model.tol,
        verbose=cfg_model.verbose,
        warm_start=cfg_model.warm_start,
        momentum=cfg_model.momentum,
        nesterovs_momentum=cfg_model.nesterovs_momentum,
        early_stopping=cfg_model.early_stopping,
        validation_fraction=cfg_model.validation_fraction,
        beta_1=cfg_model.beta_1,
        beta_2=cfg_model.beta_2,
        epsilon=cfg_model.epsilon,
        n_iter_no_change=cfg_model.n_iter_no_change,
        max_fun=cfg_model.max_fun,
    )


def add_cfg_model(cfg, tune=False):
    cfg.MLP = CN()
    cfg.MLP.hidden_layer_sizes = (300,)  # tuple, length = n_layers - 2
    cfg.MLP.activation = "logistic"  # {"identity", "logistic", "tanh", "relu"}
    cfg.MLP.solver = "adam"  # {"lbfgs", "sgd", "adam"}
    cfg.MLP.alpha = 0.0001
    cfg.MLP.batch_size = None  # int, default="auto"
    cfg.MLP.learning_rate = "adaptive"  # {"constant", "invscaling", "adaptive"}
    cfg.MLP.learning_rate_init = 0.001
    cfg.MLP.power_t = 0.5
    cfg.MLP.max_iter = 10000000000
    cfg.MLP.shuffle = True
    cfg.MLP.random_state = None  # int, default=None
    cfg.MLP.tol = 1e-4
    cfg.MLP.verbose = False
    cfg.MLP.warm_start = False
    cfg.MLP.momentum = 0.9
    cfg.MLP.nesterovs_momentum = True
    cfg.MLP.early_stopping = False
    cfg.MLP.validation_fraction = 0.1
    cfg.MLP.beta_1 = 0.9
    cfg.MLP.beta_2 = 0.999
    cfg.MLP.epsilon = 1e-8
    cfg.MLP.n_iter_no_change = 1000
    cfg.MLP.max_fun = 15000

    if tune:
        cfg_value_to_list(cfg.MLP)
