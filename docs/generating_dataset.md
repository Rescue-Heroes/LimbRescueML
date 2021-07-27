# Generating Dataset 
## Step 1: Raw Data Preparation 
Annotation of raw data should be provided as a _csv_ file following the format below:
| Filename | Label |
|:---|---:|
| session_2021-06-28-17_43_10 | 1 |
| session_2021-06-28-17_44_26 | 2 |
| session_2021-06-28-21_51_04 | 3 |

where case (both arms normal), case (left arm lymphedema), and case (right arm lymphedema) are labeled as `1`, `2`, and `3`, respectively. You can also label `0` to exclude the row. You can find an example [here](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/rawdata/annotations.csv).
Note that you should put all PGG csv files listed in column "Filename" in the same folder.

## Step 2: Data Preprocessing
[generate_dataset.py](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/generate_dataset.py) preprocesses raw data and splits it into train, validation, test datasets. 
```
usage: generate_dataset.py [-h] [--data-dir DIR] [--anno-file PATH] [--save-path PATH] [--split METHOD] [--head-drop N] [--n-samples-train N]
                           [--n-samples-test N] [--len-sample N] [--preprocess METHOD] [--switch-prob-train FLOAT] [--switch-prob-test FLOAT] [--demo PATH]
                           [--seed N]

Preprocess data and prepare datasets.

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DIR        directory of data files (default: 'rawdata/files')
  --anno-file PATH      path of annotation file (default: '')
  --save-path PATH      path (including file name) to save datesets in npz (default: 'data/dataset.npz')
  --split METHOD        the way to split train, validation, and test datesets. choices: 'random', 'random_balanced' (default, random but makes class
                        balanced), 'all_test' (use all data for test, no train and validation)
  --head-drop N         number of points to drop from head for each data file to prevent noise at the beginning (default: 150)
  --n-samples-train N   for training dataset, use 'center' to generate one center cropped sample or specify number of samples randomly generated from   each
                        date file after head dropping (default: 10)
  --n-samples-test N    for validation or test dataset, use 'center' to generate one center cropped sample or specify number of samples randomly generated
                        from each date file after head dropping (default: 'center')
  --len-sample N        number of data points for each sample (default: 300)
  --preprocess METHOD   preprocessing method to use. choices: 'normalized' (default), 'first_order', 'second_order'
  --switch-prob-train FLOAT
                        probablity to switch left and right for train. set -1 to turn off. (default: 0.5)
  --switch-prob-test FLOAT
                        probablity to switch left and right for test. set -1 to turn off, (default: -1)
  --demo PATH           a csv file to demo (default: '')
  --seed N              seed for random to re-produce the same result (default: 0)
```

**Examples:**
- simply follow the default settings:
```
python generate_dataset.py --data-dir rawdata/files --anno-file rawdata/annotations.csv --save-path data/ns10_ls300_normalized_switch.npz
```
It will generate a file "ns10_ls300_normalized_switch.npz" under "data" folder.
- set number of samples for each data file for training to 50, length for each sample to 500, switch left-right for training to off:
```
python generate_dataset.py --data-dir rawdata/files --anno-file rawdata/annotations.csv --n-samples-train 50 --len-sample 500 --switch-prob-train -1 --save-path data/ns50_ls500_normalized.npz
```
- show a demo of preprocessing for single csv file:
```
python generate_dataset.py --demo rawdata/files/session_2021-06-28-17_43_10.csv
```

<p align="center"><img src="../figures/preprocessing_demo.png" width="800"></p>

- show a demo of preprocessing for single csv file, forced to switch left and right:

```
python generate_dataset.py --demo rawdata/files/session_2021-06-28-17_43_10.csv --switch-prob-test 1
```

<p align="center"><img src="../figures/preprocessing_demo_switch.png" width="800"></p>

- dataset for test/prediction only:

```
python generate_dataset.py --data-dir rawdata/files --split all_test --save-path data/ns10_ls300_normalized_all_test.npz
```
Note that --anno-file is not specified here, which means there will be no ground truth labels. If using this dataset to our package, there will be predictions in outputs but no metrics. Specify an annotation file if needed. 