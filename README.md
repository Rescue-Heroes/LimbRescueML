# Project Limb Rescue
Cancer patients are at risk of lymphedema, a devastating chronic complication. Our overall aim is to develop a product helping patients to monitor the risk of lymphedema. The product marries wearable devices equipped with photoplethysmography (PPG) sensors and our custom software to detect changes in limb characteristics that are concerning for new-onset, or worsening lymphedema. 
Limb Rescue Cloud, constitute of Data Base, Web Tool, and Machine Learning modules, establish connections between software, doctors, and data scientists.
<img src="figures/PLR_context_diagram.png" width="500">

## Project Limb Rescue Machine Learning Module (LimbRescueML)
LimbRescueML implements four classification algorithms, Support Vector Machine(SVM), Multilayer Perceptron(MLP), Random Forest(RF), and Naive Bayes(NB) to predict lymphedema. Users can compare four classification algorithms, train and evaluate models, and predict based on saved models.

LimbRescueML provides dataset generation("generate_dataset.py"), model training("train_net.py"), model evaluation(FIXME), wave prediction(FIXME), and package installation(FIXME) pipelines.

## Installation
FIXME
## Getting Started
### Data Preprocessing
Script `generate_dataset.py` preprocesses selected rawdata, and splits preprocessed data into train, validation, test datasets.

See `python generate_dataset.py --help` for arguments options.
- Preprocessing options include: "normalized", "first_order", "second_order"
- Split methods inlcude: "random", "random_balanced"

**Example:**
```
python generate_dataset.py --data-dir DIR --anno-file PATH --save-path PATH --n-samples 30 --len-sample 100 --preprocess "normalized" --split random_balanced 
```
generates splited dataset using normalized rawdata, random balanced split method; 30 samples with wave length of 100 points are generated for each case.
### Training and Evaluation 
Script `train_net.py` is the training script. This script reads given config file for specific algorithm (including data file path, output dir and model hyperparameters, see [svm.yaml](configs/svm.yaml) as an example) to train model. Outputs include: config file backup, trained model and confusion matrice.

*Detials of config file options in [train configs](configs/README.md).*

Also, this script can tuning hyperparameters to get the best hyperparameters set using tuning configs (see [svm_tune.yaml](configs/svm_tune.yaml) as an example). Outputs include: config file backup, config file of best hyperparameters, trained model with best hyperparameters and confusion matrice.

See `python train_net.py --help` for arguments options.

**Example 1: (train model using hyperparameters in config file)**
```
python train_net.py --config-file configs/svm.yaml OUTPUT_DIR "./output_svm_1"
```
train model with hyperparameters and data specified in svm.yaml;

*Argments at the end of command line `OUTPUT_DIR "./output_svm_1"` allow overwrite config options.*

**Example 2: (tune hyperparameters)**
```
python train_net.py --config-file configs/svm_tune.yaml OUTPUT_DIR "./output_svm_2"
```
tune hyperparameters to get the best performance model settings.

### Prediction

## Model Zoo and Baselines

### Performance table
#### SVM: 
train / test accuracy: 0.82 / 0.67

confusion matrix for test set
| SVM | Pred. normal | Pred. left | Pred. right |
| --- | --- | --- | --- | 
| True normal | 20 | 0 | 0 |
| True left | 10 | 10 | 0 |
| True right | 10 | 0 | 10 |

#### MLP: 
train / test accuracy: 0.82 / 0.67

confusion matrix for test set
| MLP | Pred. normal | Pred. left | Pred. right |
| --- | --- | --- | --- | 
| True normal | 20 | 0 | 0 |
| True left | 10 | 10 | 0 |
| True right | 10 | 0 | 10 |

#### RF: 
train / test accuracy: 1.00 / 0.67

confusion matrix for test set
| RF | Pred. normal | Pred. left | Pred. right |
| --- | --- | --- | --- | 
| **True normal** | 18.8 (0.94) | 0 (0.00) | 0 (0.00) |
| **True left** | 8.4 (0.42) | 11.6 (0.58) | 0 (0.00) |
| **True right** | 10.4 (0.52) | 0 (0.00) | 9.6 (0.48) |


## People
Sponsors: Carlo Contreras, Lynne Brophy
Technical Team: 
- [Tai-Yu Pan](https://github.com/tydpan) implemented dataset generation pipeline, model trainning and evaluation pipeline
- [Mengdi Fan](https://github.com/mengdifan) implemented model training and evaluation pipeline
- [Rithvich Ramesh](https://github.com/rithvichramesh) tested the gaussian naive bayes algorithm
- [Browy Li](https://github.com/BrowyLi)


