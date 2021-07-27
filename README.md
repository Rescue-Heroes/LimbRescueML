# Project Limb Rescue
Cancer patients are at risk of lymphedema, a devastating chronic complication. Our overall aim is to develop a product helping patients to monitor the risk of lymphedema. The product marries wearable devices equipped with photoplethysmography (PPG) sensors and our custom software to detect changes in limb characteristics that are concerning for new-onset, or worsening lymphedema. 
Limb Rescue Cloud, constitute of Data Base, Web Tool, and Machine Learning modules, establish connections between software, doctors, and data scientists.
<p align="center"><img src="figures/PLR_context_diagram.png" width="400"></p>

## Project Limb Rescue Machine Learning Module (LimbRescueML)
LimbRescueML implements four classification algorithms, Support Vector Machine(SVM), Multilayer Perceptron(MLP), Random Forest(RF), and Naive Bayes(NB) to predict lymphedema. Users can compare four classification algorithms, train and evaluate models, and predict with saved models.

LimbRescueML provides dataset generation([Generating Dataset](https://github.com/Rescue-Heroes/LimbRescueML#getting-started)), model training([Training](https://github.com/Rescue-Heroes/LimbRescueML#training)), model evaluation([Evaluation](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/evaluation.md)), wave prediction([Prediction](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/prediction.md) for more information.), and package installation(FIXME) pipelines.

## Installation
- We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to manage packages and dependencies. If it is not installed on your machine, follow the instruction on their website. 
- clone our repository:
```
git clone git@github.com:Rescue-Heroes/LimbRescueML.git
cd LimbRescueML
```
- create enviornment and install dependencies:
```
conda env create --name LimbResCloud --file environment.yaml
conda activate LimbResCloud
```
If you have already had LimbResCloud in your environment, run the following instead:
```
conda env update --name test --file environment.yaml
conda activate LimbResCloud
```
- install our machine learning package:
```
python -m pip install -e .
```
- check if it's installed correctly. You should find "limbresml" on the package list with the following command:
```
conda list
```
- After installation, remember to activate the enviornment whenever you want to use the package. 
```
conda activate LimbResCloud
```

## Getting Started
### Generating Dataset 
- Step 1: prepare raw data sets and corresponding annotation file. See [Raw Data Preparation](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/generating_dataset.md#raw-data-preparation) for more information.
- Step 2: generate preprocessed datasets for model training and wave prediction. See [Data Preprocessing](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/generating_dataset.md#data-preprocessing) for more information.

### Training
Training pipeline provides two functionalities:
- Training model with specific hyperparameters set. Trained model and prediction table will be saved. See [Training Model](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/training.md#training-model-with-hyperparameters-specified) for more information.
- Tuning hyperparameters to find the best model settings. Best hyperparameters set, trained model and prediction table will be saved. See [Tuning hyperparameters](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/training.md#tuning-hyperparameters-for-best-model-settings) for more information.

### Evaluation
Evlauation pipline allows user to evaluate saved trained models with test dataset or datasets from labeled new raw data. See [Evaluation](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/evaluation.md) for more information.

### Prediction
Prediction pipline allows user to make predictions for unlabeled new raw data based on trained model. Raw data preprocess should follow [Generating Dataset](https://github.com/Rescue-Heroes/LimbRescueML#getting-started). See [Prediction](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/prediction.md) for more information.

## Model Zoo and Baselines
_NOTE: The following performance table and confusion matrices are generated based on raw data provided by July 19th, 2021._

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
- [Tai-Yu Pan](https://github.com/tydpan) implemented dataset generation pipeline, model training and evaluation pipeline
- [Mengdi Fan](https://github.com/mengdifan) implemented model training and evaluation pipeline, generated documentation
- [Rithvich Ramesh](https://github.com/rithvichramesh) tested the Gaussian Naive Bayes algorithm
- [Browy Li](https://github.com/BrowyLi) tested the Random Forest algorithm


