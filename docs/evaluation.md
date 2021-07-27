# Evaluation

Script [train_net.py](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/train_net.py) is the main script.

- Step 1: Modify config file to specify the dataset path, output directory, **saved model** and other settings (see [svm.yaml](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/configs/svm.yaml) as an example).

    **MODEL** in config file should be set as the path of saved model._Details of config file options in [train configs](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/configs/README.md)._

- Step 2: Evaluate model using script [train_net.py](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/train_net.py). **Argument `--eval-only` must be included to evaluate saved model.**

    This script reads the given config file(path to saved model) for a specific algorithm to train model. Outputs include: config file backup(_yaml_), accuracy(_txt_), predictions table(_csv_) and confusion matrice(_png_).

    See `python train_net.py --help` for arguments options. Script details can be found in [docs/train_net.md](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/train_net.md).

    **Example: **

    ```
    python train_net.py --eval-only  --config-file configs/svm.yaml  MODEL demo_saved_model/SVM.joblib INPUT.PATH data/ns20_ls300_normalized.npz
    ```
    Above command: evaluate saved model with different dataset (samples should have same length, if training using dataset with length of 100, the model can be only used for other dataset with length of 100); outputs are saved to output path in config file.

    _Arguments at the end of command line `OUTPUT_DIR "./output_svm_1"` allow overwriting config options. Users can either directly modify config file or add arguments at the end of command line to overwrite values in config file._