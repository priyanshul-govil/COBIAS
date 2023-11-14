# COBIAS

This repository contains the dataset, metric, and evaluation data and results presented in "COBIAS: Leveraging Context for Bias Assessment" submitted to TheWebConf 2024. This repository will be updated with additional details post acceptance of our manuscript.

## Directory Structure

* `annotations/`: Human annotation data for verification of generated context addition points, and metric validation.
* `datasets/`: `COBIAS.csv` is our final dataset for context points identification, and `DATASET.csv` is the intermediate data created for human verification.
* `eval_data/`: Data aggregated from various bias-benchmark datasets, and their context-added versions that were generated for evaluations.
* `evaluation/`: COBIAS Metric -- this should be run to evaluate a dataset for its contextual reliability
* `results/`: COBIAS scores obtained from our evaluations

## Running COBIAS on a dataset

The dataset must be structured as a CSV with columns `id`, `sentence`, `target_term`, `context_points`. Refer to `datasets/COBIAS.csv` for the structure. Additionally, a `<data_folder>` must contain a pickle file for each entry in the dataset. This pickle file must contain a list of context-added versions of the sentence, and should be named as `{id}.pkl`.

### Installing dependencies

```
python3 -m pip install -r requirements.txt
```

### Running COBIAS

```
python3 evaluation/scorer.py --input_file <path_to_dataset.csv> --output_file <results/output_file.csv> --data_folder <path_to_data_folder/>
```