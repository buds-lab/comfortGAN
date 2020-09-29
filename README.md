<!-- For the public repo use the following badges:
[![DOI:10.1007/978-3-319-76207-4_15](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1007/978-3-319-76207-4_15)
 -->
# Balancing thermal comfort datasets: We GAN, but should we?
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)  ![Python Version](https://upload.wikimedia.org/wikipedia/commons/f/fc/Blue_Python_3.7_Shield_Badge.svg) ![PyTorch version](https://img.shields.io/badge/PyTorch-1.6-blue)

This repository is the official implementation of [Balancing thermal comfort datasets: We GAN, but should we?](https://arxiv.org/abs/2009.13154).

## Requirements

### Python
```setup
conda env create --file envs/environment_<OS>.yaml # replace OS with either `macos` or `ubuntu`
```
Other environments used for running the benchmarks can be found in this folder.

### Datasets

The datasets used are:

- [Field experiment](https://github.com/buds-lab/humans-as-a-sensor-for-buildings)

- [Controlled experiment](https://doi.org/10.5281/zenodo.3363987)

- [ASHRAE Global Thermal Comfort Database II](https://datadryad.org/stash/dataset/doi:10.6078/D1F671)

Once the datasets are downloaded into `data/`, the notebook `1-DatasetsPreparation.ipynb` performs the feature selection specified on the paper and creates the train and test sets for each dataset. Then the notebooks `2`,`3`, and `4` calculates the baseline metrics for the datasets respectively. Notebook `2` and `4` has an `-a` version which treats the dataset as is, with their original number classes, and a `-b` version where the classes are remapped to only 3 classes.

## Training

To train the model(s) in the paper, run this command: 

```train
python train.py configs/config_comfortgan_occutherm_128-1-20.json 
```

where `config_comfortgan_<dataset>_<batch_size>-<n_critic>-<latent_space>.json` is a configuration file with information regarding the hyperparameters of the model to be trained. The different configuration files tested on this paper are located in `configs/`. Configuration files for the `reduced` datasets have the wording `-reduced` after the dataset name: E.g., `config_comfortgan_occutherm-reduced_128-1-20.json`.


In order to train our model in all datasets, run:

```train all classes
./run_train.sh
```

Or, for the datasets reduced to only 3 classes:

```train reduced classes
./run_train-reduced.sh
```

Our pre-trained models can be found in `models/` and the losses in `tensorboard/loss/`.

## Evaluation

To evaluate our model(s), run:

```eval 
python evaluation_trials.py configs/config_comfortgan_occutherm_128-1-20.json 
```

Or for the reduced dataset:

```eval 
python evaluation_trials-reduced.py configs/config_comfortgan_occutherm-reduced_128-1-20.json 
```

The configuration files are the same used for training in the previous section.

To evaluate on all datasets, run:

```eval all classes
./run_evaluations.sh
```

Or, on all datasets reduced to only 3 classes:

```eval reduced classes
./run_evaluations-reduced.sh
```

The results can be visualize in notebooks `8`.

## Benchmmark

The notebooks `5`, `6`, and `7` are used for training and evaluating other models used for benchmarking, with an `a` version for datasets with all classses and `b` for the datasets with reduced classes. Auxiliary files are found in `benchmarks/`.

These evaluations results can be visualized in notebooks `8a` for datastes with all classes and `8b` for the datasets with reduced classes.

## Results

Our model achieves the following performance (`micro F1-score`) on the three publicly avaialable datasets, the results of two commonly used augmentation methods in the related literature are shown too:

| Model                        | Controlled <br /> Dataset | Field <br /> Dataset | ASHRAE Global Thermal <br /> Comfort Database II |
| ---------------------------- | :-----------------------: | :------------------: | :----------------------------------------------: |
| Baseline                     |            0.60           |         0.65         |                       0.26                       |
| SMOTE                        |            0.47           |         0.53         |                       0.30                       |
| ADASYN                       |            0.41           |         0.51         |                       0.30                       |
| comfortGAN                   |            0.64           |         0.67         |                       0.43                       |
| Baseline <br /> (3 classes)  |            0.66           |         0.65         |                       0.49                       |
| SMOTE <br /> (3 classes)     |            0.60           |         0.53         |                       0.50                       |
| ADASYN <br /> (3 classes)    |            0.53           |         0.51         |                       0.50                       |
| comfortGAN <br /> (3 classes)|            0.72           |         0.67         |                       0.51                       |


The original table can be found on the paper (Table 1).

## Visualization

Plots used in the paper can be reproduced with notebook `9`.

## Contributing

MIT License
