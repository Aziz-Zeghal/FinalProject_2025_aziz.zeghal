# Short Video Recommender System (KuaiRec)

This repository contains the implementation of a short video recommender system with the KuaiRec dataset.

---

## **Table of Contents**

- [Setup](#setup)
- [Dataset](#dataset)
- [Models](#models)
    - [Model 1: ALS](#model-1-als)

## Setup
Set up the environment using Conda:
```bash
conda env create -f conda_env.yaml
```
Inside the Jupyter Notebook, select the environment `REMA1` as the kernel.

Export the environment:
```bash
conda env export --name REMA1 > conda_env.yaml
```

## Dataset
The dataset used in this project is the KuaiRec dataset, which contains user interactions with short videos. The user-item interactions are stored in the `small_matrix.csv` (99.6% density) and the `big_matrix.csv` (16.3% density) files.

For more information about the dataset, please refer to the [KuaiRec dataset website](https://kuairec.com/).

## Models
Multiple models were tested for this project.

### Model 1: ALS
The Alternating Least Squares (ALS) model was initialy implemented using the `implicit` library, and later changed to `pyspark.ml.recommendation` for better scalability. This library enables hyperparameter tuning and model evaluation using cross-validation.

I encountered problems training the ALS model only with the `small_matrix.csv` dataset. So I used the sparse dataset `big_matrix.csv` instead. 