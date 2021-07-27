# Training

Script [train_net.py](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/train_net.py) is the main script.

## Training model with hyperparameters specified

- Step 1: Modify config file to specify the dataset path, output directory, model hyperparameters and other settings (see [svm.yaml](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/configs/svm.yaml) as an example).

    _Details of config file options in [train configs](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/configs/README.md)._

- Step 2: Train model using script [train_net.py](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/train_net.py).

    This script reads the given config file for a specific algorithm to train model. Outputs include: config file backup(_yaml_), trained model(_joblib_), accuracy(_txt_), predictions table(_csv_) and confusion matrice(_png_).

    See `python train_net.py --help` for arguments options. Script details can be found in [docs/train_net.md](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/train_net.md).

    **Examples: (train model using hyperparameters in config file)**

    ```
    python train_net.py --config-file configs/svm.yaml
    ```
    Above command: train model with hyperparameters and dataset specified in `configs/svm.yaml`; outputs are saved to output path in config file.

    ```
    python train_net.py --config-file configs/svm.yaml OUTPUT_DIR "./output_svm_1" SVM.C 6.0
    ```
    Above command: train model with hyperparameters and dataset specified in `configs/svm.yaml`; outputs are saved to `./output_svm_1` instead of the default `OUTPUT_DIR` in `svm.yaml`; hyperparameter `C` in `SVM` algorithm is changed to `6.0`. 

    _Arguments at the end of command line `OUTPUT_DIR "./output_svm_1"` allow overwriting config options. Users can either directly modify config file or add arguments at the end of command line to overwrite values in config file._

## Tuning hyperparameters for best model settings

- Step 1: Modify config file to specify the dataset path, output directory, model hyperparameters choices and other settings (see [svm_tune.yaml](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/configs/svm_tune.yaml) as an example).

    _Details of config file options in [train configs](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/configs/README.md)._

- Step 2: Tune hyperparameters using script [train_net.py](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/train_net.py).

    Also, this script can tune hyperparameters to get the best hyperparameters set using tuning configs. Outputs include: config file backup(_yaml_), config file of best hyperparameters(_yaml_), trained model with best hyperparameters(_joblib_), predictions table(_csv_) and confusion matrice(_png_).

    **Examples: (tune hyperparameters)**
    ```
    python train_net.py --config-file configs/svm_tune.yaml
    ```
    Above command: tune hyperparameters to get model settings with the best validation dataset performance, using hyperparameters choices in `configs/svm_tune.yaml`; outputs are saved to output path in config file.

    ```
    python train_net.py --config-file configs/svm_tune.yaml OUTPUT_DIR "./output_svm_2" SVM.C "[2.0, 6.0]"
    ```
    Above command: tune hyperparameters to get model settings with the best validation dataset performance, using hyperparameters choices in `configs/svm_tune.yaml`; outputs are saved to `./output_svm_1` instead of the default `OUTPUT_DIR` in `svm_tune.yaml`; test choices of hyperparameter `C` in `SVM` algorithm are changed to `[2.0, 6.0]` instead of choices list in `svm_tune.yaml`(`[2.0, 6.0]` must in `""`).