# Dataset Generation
## Raw Data Preparation 
## Data Preprocessing
Script [generate_dataset.py](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/generate_dataset.py) preprocesses selected raw data with corresponding annotation file([example](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/rawdata/annotations.csv) and [requirements](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/dataset_generation.md#raw-data-preparation)), and splits preprocessed data into train, validation, test datasets. Outputs include: dataset npz file.

See `python generate_dataset.py --help` for arguments options. Script details can be found in [docs/generate_dataset.md](https://github.com/Rescue-Heroes/LimbRescueML/blob/main/docs/generate_dataset.md).
- Preprocessing options include: `"normalized"`, `"first_order"`, `"second_order"`
- Split methods inlcude: `"random"`, `"random_balanced"`

**Example:**
```
python generate_dataset.py --data-dir DIR --anno-file PATH --save-path PATH --n-samples 30 --len-sample 100 --preprocess "normalized" --split random_balanced 
```
Above command generates a split dataset using normalized raw data, random balanced split method; 30 samples with the wavelength of 100 points are generated for each raw data case.