# Project Limb Rescue
Cancer patients are at risk of lymphedema, a devastating chronic complication. Our overall aim is to develop a product helping patients to monitor the risk of lymphedema. The product marries wearable devices equipped with photoplethysmography (PPG) sensors and our custom software to detect changes in limb characteristics that are concerning for new-onset, or worsening lymphedema. 
Limb Rescue Cloud, constitute of Data Base, Web Tool, and Machine Learning modules, establish connections between software, doctors, and data scientists.
![alt text](figures/PLR_context_diagram.png "PLR conext diagram")
## Project Limb Rescue Machine Learning Module (LimbRescueML)
LimbRescueML implements four classification algorithms, support vector machine(svm), multilayer perceptron(mlp), random forest(rf), and naive bayes(nb) to predict lymphedema. Users can compare four classification algorithms, train and evaluate models, and predict based on saved models.

LimbRescueML provides dataset generation("generate_dataset.py"), model training("train_net.py"), model evaluation(FIXME), wave prediction(FIXME), and package installation(FIXME) pipelines.

## Installation

## Getting Started
### Data Preprocessing
Script "generate_dataset.py" preprocesses selected rawdata, and splits preprocessed data into train, validation, test datasets.\
See `python generate_dataset.py --help` for arguments options.\
- Preprocessing options include: "normalized", "first_order", "second_order"\
- Split methods inlcude: "random", "random_balanced"\
Example:
`python generate_dataset.py --split random_balanced --save-path PATH --n-samples 30 --len-sample 100 --preprocess "normalized"`\
generates splited dataset using normalized rawdata, random balanced split method; 30 samples with wave length of 100 points are generated for each case.


### Training and Evaluation 
Script "train_net.py" 

### Prediction

## Model Zoo and Baselines

## People
Sponsors: Carlo Contreras, Lynne Brophy
Technical Team: 
- [Tai-Yu Pan](https://github.com/tydpan) implemented dataset generation pipeline, model trainning and evaluation pipeline
- [Mengdi Fan](https://github.com/mengdifan) implemented model trainning and evaluation pipeline
- [Rithvich Ramesh](https://github.com/rithvichramesh)
- [Browy Li](https://github.com/BrowyLi)


