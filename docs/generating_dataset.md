# Generating Dataset 
## Step 1: Raw Data Preparation 
Annotation of raw data should be provided as a _csv_ file following format as below:
| Filename | Label | Blood pressure cuff laterality | Inflation (mmHg) | Comments |
|:---|---:|---:|---:|---:|
| session_2021-06-28-17_43_10 | 1 | none | | |
| session_2021-06-28-17_44_26 | 2 | left | 60 | |
| session_2021-06-28-21_51_04 | 3 | right | 60 | |

where case (both arms normal), case (left arm lymphedema), and case (right arm lymphedema) are labeled as `1`, `2`, and `3`, respectively; cases with only one arm data should be eliminated in the annotation file.

[Example](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/rawdata/annotations.csv) of Annotation file.

## Step 2: Data Preprocessing
Script [generate_dataset.py](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/generate_dataset.py) preprocesses selected raw data with corresponding annotation file([example](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/rawdata/annotations.csv) and [requirements](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/generating_dataset.md#raw-data-preparation)), and splits preprocessed data into train, validation, test datasets. Outputs include: dataset npz file.

See `python generate_dataset.py --help` for arguments options. Script details can be found in [docs/generate_dataset.md](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/generate_dataset.md).
- Preprocessing options include: `"normalized"`, `"first_order"`, `"second_order"`
- Split methods inlcude: `"random"`, `"random_balanced"`

**Example:**
```
python generate_dataset.py --data-dir DIR --anno-file PATH --save-path PATH --n-samples 30 --len-sample 100 --preprocess "normalized" --split random_balanced 
```
Above command generates a split dataset using normalized raw data, random balanced split method; 30 samples with the wavelength of 100 points are generated for each raw data case.