# Project Limb Rescue
Cancer patients are at risk of lymphedema, a devastating chronic complication. Our overall aim is to develop a product helping patients to monitor the risk of lymphedema. The product marries wearable devices equipped with photoplethysmography (PPG) sensors and our custom software to detect changes in limb characteristics that are concerning for new-onset, or worsening lymphedema. 
Limb Rescue Cloud, constitute of Data Base, Web Tool, and Machine Learning modules, establish connections between software, doctors, and data scientists.
<p align="center"><img src="figures/PLR_context_diagram.png" width="400"></p>

## Project Limb Rescue Machine Learning Module (LimbRescueML)
LimbRescueML implements four classification algorithms, Support Vector Machine(SVM), Multilayer Perceptron(MLP), Random Forest(RF), and Naive Bayes(NB) to predict lymphedema. Users can compare four classification algorithms, train and evaluate models, and predict based on saved models.

LimbRescueML provides dataset generation([`generate_dataset.py`](generate_dataset.py)), model training([`"train_net.py"`](train_net.py)), model evaluation(FIXME), wave prediction(FIXME), and package installation(FIXME) pipelines.

## Installation
FIXME
## Getting Started
### Data Preprocessing
Script [`generate_dataset.py`](generate_dataset.py) preprocesses selected raw data, and splits preprocessed data into train, validation, test datasets. Outputs include: dataset npz file.

See `python generate_dataset.py --help` for arguments options. Script details can be found in [`docs/generate_dataset.md`](docs/generate_dataset.md).
- Preprocessing options include: `"normalized"`, `"first_order"`, `"second_order"`
- Split methods inlcude: `"random"`, `"random_balanced"`

**Example:**
```
python generate_dataset.py --data-dir DIR --anno-file PATH --save-path PATH --n-samples 30 --len-sample 100 --preprocess "normalized" --split random_balanced 
```
Above command generates a split dataset using normalized raw data, random balanced split method; 30 samples with the wavelength of 100 points are generated for each raw data case.

### Training
Script [`train_net.py`](train_net.py) is the training script. This script reads the given config file for a specific algorithm (including data file path, output dir and model hyperparameters, see [svm.yaml](configs/svm.yaml) as an example) to train model. Outputs include: config file backup(`yaml`), trained model(`joblib`), accuracy(`txt`) and confusion matrice(`png`).

*Details of config file options in [train configs](configs/README.md).*

Also, this script can tune hyperparameters to get the best hyperparameters set using tuning configs (see [`svm_tune.yaml`](configs/svm_tune.yaml) as an example). Outputs include: config file backup(`yaml`), config file of best hyperparameters(`yaml`), trained model with best hyperparameters(`joblib`) and confusion matrice(`png`).

See `python train_net.py --help` for arguments options. Script details can be found in [`docs/train_net.md`](docs/train_net.md).

**Example 1: (train model using hyperparameters in config file)**
```
python train_net.py --config-file configs/svm.yaml OUTPUT_DIR "./output_svm_1 SVM.C 6.0"
```
Above command: train model with hyperparameters and dataset specified in `svm.yaml`; outputs are saved to `./output_svm_1` instead of the default `OUTPUT_DIR` in `svm.yaml`; hyperparameter `C`in `SVM` algorithm is changed to `6`. 

*Arguments at the end of command line `OUTPUT_DIR "./output_svm_1"` allow overwriting config options.*

**Example 2: (tune hyperparameters)**
```
python train_net.py --config-file configs/svm_tune.yaml OUTPUT_DIR "./output_svm_2"
```
Above command: tune hyperparameters to get model settings with the best validation dataset performance, using hyperparameters choices in `svm_tune.yaml`.

### Evaluation
FIXME
### Prediction
FIXME
## Model Zoo and Baselines
*NOTE: The following performance table and confusion matrices are generated based on raw data provided by July 19th, 2021.*

We performed five tuning hyperparameters runs and picked the hyperparameters set with the highest validation accuracy as default (default yaml configs) for each algorithm. Then we trained each algorithm with these default hyperparameters set five times and averaged the accuracy and confusion matrix as follows.

### Performance table
| Accuracy | Train | Test |
|:---|---:|---:|
| SVM | 0.82 | 0.67 |
| MLP | 0.92 | 0.58 |
| RF | 1.00 | 0.67 |
| NB | 0.62 | 0.58 |

### Confusion matrix for test set
| SVM | Pred. normal | Pred. left | Pred. right |
| :--- | ---: | ---: | ---: | 
| **True normal** | 20.0 (1.00) | 0.0 (0.00) | 0.0 (0.00) |
| **True left** | 10.0 (0.50) | 10.0 (0.50) | 0.0 (0.00) |
| **True right** | 10.0 (0.50) | 0.0 (0.00) | 10.0 (0.50) |

| MLP | Pred. normal | Pred. left | Pred. right |
| :--- | ---: | ---: | ---: | 
| **True normal** | 11.6 (0.58) | 0.0 (0.00) | 8.4 (0.42) |
| **True left** | 5.0 (0.25) | 14.0 (0.70) | 1.0 (0.05) |
| **True right** | 10.0 (0.50) | 0.8 (0.04) | 9.2 (0.46) |

| RF | Pred. normal | Pred. left | Pred. right |
| :--- | ---: | ---: | ---: |
| **True normal** | 18.8 (0.94) | 0.0 (0.00) | 0.0 (0.00) |
| **True left** | 8.4 (0.42) | 11.6 (0.58) | 0.0 (0.00) |
| **True right** | 10.4 (0.52) | 0.0 (0.00) | 9.6 (0.48) |

| NB | Pred. normal | Pred. left | Pred. right |
| :--- | ---: | ---: | ---: |
| **True normal** | 20.0 (1.00) | 0.0 (0.00) | 0.0 (0.00) |
| **True left** | 15.0 (0.75) | 5.0 (0.25) | 0.0 (0.00) |
| **True right** | 10.0 (0.50) | 0.0 (0.00) | 10.0 (0.50) |

## People
Sponsors: Carlo Contreras, Lynne Brophy

Technical Team: 
- [Tai-Yu Pan](https://github.com/tydpan) implemented dataset generation pipeline, model trainning and evaluation pipeline
- [Mengdi Fan](https://github.com/mengdifan) implemented model training and evaluation pipeline, generated documetation
- [Rithvich Ramesh](https://github.com/rithvichramesh) tested the gaussian naive bayes algorithm
- [Browy Li](https://github.com/BrowyLi) tested the random forest algorithm


