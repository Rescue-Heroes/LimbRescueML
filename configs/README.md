### Training configs
Specify the algorithm/model, hyperparameters, input and output path in config file; will be used in training pipline.

Taking `svm.yaml` as an example:
- MODEL: String type algorithm name, including `SVM`, `MLP`, `NB`, and `RF`, must be string type ("SVM"); should be `"SVM"` for  `svm.yaml`.
- TUNE_HP_PARAMS: Boolean type flag, indicates whether to tune hyperparameters or not; should be `False` for training config.
- SVM: contains all hyperparameters setted by user; each hyperparameter must follow the default type, for example, `C: 7.0` C should be a float type value, use `7.0` instead of `7`; `degree: 1` degree should be a int type value, use `1` instead of `1.0`.
- INPUT:
    - PATH: String type data file name, must be sting type("datafile").
    - CAT_TRAIN_VAL: Boolean type flag, indicates whether to combine train and validation dataset; normally, validation dataset is used in model tuning process; should be `True` for training config.
- OUTPUT_DIR: String type output file path, must be sting type("output_dir").


### Tuning hyperparameters configs
Specify the algorithm/model, hyperparameters tuning choices, input and output path in config file; will be used in training pipline.

Taking `svm_tune.yaml` as an example:
- MODEL: String type algorithm name, including `SVM`, `MLP`, `NB`, and `RF`, must be string type ("SVM"); should be `"SVM"` for  `svm_tune.yaml`.
- TUNE_HP_PARAMS: Boolean type flag, indicates whether to tune hyperparameters or not; should be `True` for tuning config.
- SVM: contains all hyperparameters choices; choices should be listed in `[]`; each hyperparameter must follow the default type, for example, `C: [7.0]` C should be a float type value, use `7.0` instead of `7`; `degree: [1]` degree should be a int type value, use `1` instead of `1.0`.
- INPUT:
    - PATH: String type data file name, must be sting type("datafile").
    - CAT_TRAIN_VAL: Boolean type flag, indicates whether to combine train and validation dataset; normally, validation dataset is used in model tuning process; should be `False` for training config.
- OUTPUT_DIR: String type output file path, must be sting type("output_dir").