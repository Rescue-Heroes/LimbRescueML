### Training configs
Specify the algorithm/model, hyperparameters, input, and output path in the config file; will be used in the training pipeline.

Taking [svm.yaml](svm.yaml) as an example:
- **MODEL**: String type algorithm name, including _SVM_, _MLP_, _NB_, and _RF_, must be string type (`"SVM"`); should be `"SVM"` for [svm.yaml](svm.yaml).
- **TUNE_HP_PARAMS**: Boolean type flag, indicates whether to tune hyperparameters or not; should be `False` for training config.
- **SVM**: contains all hyperparameters set by users; each hyperparameter must follow the default type, for example, `C: 7.0` `C` should be a float type value, use `7.0` instead of `7`; `degree: 1` degree should be an int type value, use `1` instead of `1.0`.
- **INPUT**:
    - **PATH**: String type dataset path, must be sting type(`"datafile.npz"`).
    - **CAT_TRAIN_VAL**: Boolean type flag, indicates whether to combine train and validation dataset; normally, validation dataset is used in model tuning process; should be `True` for training config.
- **OUTPUT_DIR**: String type output file path, must be sting type(`"output_dir"`).

_Defaul hyperparameters are provided based on test runs._


### Tuning hyperparameters configs
Specify the algorithm/model, hyperparameters tuning choices, input, and output path in config file; will be used in training pipeline.

Taking [svm_tune.yaml](svm_tune.yaml) as an example:
- **MODEL**: String type algorithm name, including _SVM_, _MLP_, _NB_, and _RF_, must be string type (`"SVM"`); should be `"SVM"` for  [svm_tune.yaml](svm_tune.yaml).
- **TUNE_HP_PARAMS**: Boolean type flag, indicates whether to tune hyperparameters or not; should be `True` for tuning config.
- **SVM**: contains all hyperparameters choices; choices should be listed in `[]`; each hyperparameter must follow the default type, for example, `C: [7.0]` `C` should be a float type value, use `7.0` instead of `7`; `degree: [1]` degree should be an int type value, use `1` instead of `1.0`.
- **INPUT**:
    - **PATH**: String type dataset path, must be sting type(`"datafile.npz"`).
    - **CAT_TRAIN_VAL**: Boolean type flag, indicates whether to combine train and validation dataset; normally, validation dataset is used in model tuning process; should be `False` for training config.
- **OUTPUT_DIR**: String type output file path, must be sting type(`"output_dir"`).


*Note: all values can be changed either in the config file or via the command line, for example, adding `SVM.C 5.0` to the end of the command line when running the script will use `C` value at `5.0` no matter what value it is in the config file.* 