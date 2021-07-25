from .config import CfgNode as CN


def default_cfg(model_name):
    _C = CN()

    _C.TUNE_HP_PARAMS = False

    if model_name == "SVM":
        # MODEL
        _C.MODEL = "SVM"
        # -----------------------------------------------------------------------------
        # Support Vector Machine
        # For full documentation, please see:
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        # -----------------------------------------------------------------------------
        _C.SVM = CN()
        _C.SVM.C = 7.0
        _C.SVM.kernel = "rbf"  # {"linear", "poly", "rbf", "sigmoid", "precomputed"}
        _C.SVM.degree = 1
        _C.SVM.gamma = None  # {"scale", "auto"} or float, default="scale"
        _C.SVM.coef0 = 0.0
        _C.SVM.shrinking = True
        _C.SVM.probability = False
        _C.SVM.tol = 1e-3
        _C.SVM.cache_size = 200.0
        _C.SVM.class_weight = None  # dict or "balanced", default=None
        _C.SVM.verbose = False
        _C.SVM.max_iter = -1
        _C.SVM.decision_function_shape = "ovo"  # {"ovo", "ovr"}
        _C.SVM.break_ties = False
        _C.SVM.random_state = None  # None or int

    if model_name == "MLP":
        # MODEL
        _C.MODEL = "MLP"
        # -----------------------------------------------------------------------------
        # Multilayer Perceptron
        # For full documentation, please see:
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        # -----------------------------------------------------------------------------
        _C.MLP = CN()
        _C.MLP.hidden_layer_sizes = (300,)  # tuple, length = n_layers - 2
        _C.MLP.activation = "logistic"  # {"identity", "logistic", "tanh", "relu"}
        _C.MLP.solver = "adam"  # {"lbfgs", "sgd", "adam"}
        _C.MLP.alpha = 0.0001
        _C.MLP.batch_size = None  # int, default="auto"
        _C.MLP.learning_rate = "adaptive"  # {"constant", "invscaling", "adaptive"}
        _C.MLP.learning_rate_init = 0.001
        _C.MLP.power_t = 0.5
        _C.MLP.max_iter = 10000000000
        _C.MLP.shuffle = True
        _C.MLP.random_state = None  # int, default=None
        _C.MLP.tol = 1e-4
        _C.MLP.verbose = False
        _C.MLP.warm_start = False
        _C.MLP.momentum = 0.9
        _C.MLP.nesterovs_momentum = True
        _C.MLP.early_stopping = False
        _C.MLP.validation_fraction = 0.1
        _C.MLP.beta_1 = 0.9
        _C.MLP.beta_2 = 0.999
        _C.MLP.epsilon = 1e-8
        _C.MLP.n_iter_no_change = 1000
        _C.MLP.max_fun = 15000

    if model_name == "RF":
        # MODEL
        _C.MODEL = "RF"
        # -----------------------------------------------------------------------------
        # Random Forest
        # For full documentation, please see:
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        # -----------------------------------------------------------------------------
        _C.RF = CN()
        _C.RF.n_estimators = 10
        _C.RF.criterion = "entropy"  # {"gini", "entropy"}
        _C.RF.max_depth = None  # None or int
        _C.RF.min_samples_split = None  # int or float, default=2
        _C.RF.min_samples_leaf = None  # int or float, default=1
        _C.RF.min_weight_fraction_leaf = 0.0
        _C.RF.max_features = None  # {"auto", "sqrt", "log2"}, int or float, default="auto"
        _C.RF.max_leaf_nodes = None  # None or int
        _C.RF.min_impurity_decrease = 0.0
        _C.RF.min_impurity_split = None  # None or float
        _C.RF.bootstrap = True
        _C.RF.oob_score = False
        _C.RF.n_jobs = None  # None or int
        _C.RF.random_state = None  # None or int
        _C.RF.verbose = 0
        _C.RF.warm_start = False
        _C.RF.class_weight = None  # {None, "balanced", "balanced_subsample"}, dict or list of dicts
        _C.RF.ccp_alpha = 0.2
        _C.RF.max_samples = None  # None, int or float

    if model_name == "NB":
        # MODEL
        _C.MODEL = "NB"
        # -----------------------------------------------------------------------------
        # Gaussian Naive Bayes
        # For full documentation, please see:
        # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        # -----------------------------------------------------------------------------
        _C.NB = CN()
        _C.NB.priors = None  # array-like of shape(n_classes,)
        _C.NB.var_smoothing = 1e-08

    # -----------------------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------------------
    _C.INPUT = CN()
    _C.INPUT.PATH = "data/ns10_ls300_normalized.npz"
    _C.INPUT.CAT_TRAIN_VAL = True

    # -----------------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------------
    _C.OUTPUT_DIR = "./output"
    return _C
