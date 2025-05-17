# Short Video Recommender System (KuaiRec)

This repository contains the implementation of different short video recommender systems with the KuaiRec dataset.

The main focus of the project was the *Two-Tower Neural Network* model.
---

## **Table of Contents**

- [Setup](#setup)
- [Dataset](#dataset)
- [EDA](#eda)
- [Models](#models)
    - [Model 1: ALS](#model-1-als)
    - [Model 2: Two-Tower Neural Network](#model-2-two-tower-neural-network)
    - [Model 3: Cosine Similarity](#model-3-cosine-similarity)

## Setup
### Environment
Set up the environment using Conda:
```bash
conda env create -f conda_env.yaml
```
Inside the Jupyter Notebook, select the environment `REMA1` as the kernel.

Export the environment:
```bash
conda env export --name REMA1 > conda_env.yaml
```
### Downloading the dataset
You would want to download the dataset from the [KuaiRec dataset website](https://kuairec.com/). Place it in the root folder, or in the `models` folder.

I used the google drive dataset. The folder is named `KuaiRec 2.0`.

## Dataset
The dataset used in this project is the KuaiRec dataset, which contains user interactions with short videos. The user-item interactions are stored in the `small_matrix.csv` (99.6% density) and the `big_matrix.csv` (16.3% density) files.

For more information about the dataset, please refer to the [KuaiRec dataset website](https://kuairec.com/).

## EDA
The exploratory data analysis was performed on all the elements of the dataset.

For more detail, you can check the `EDA.ipynb` notebook.

## Models
Multiple models were tested for this project. They are listed in chronological order of implementation in the report.

### Model 1: ALS
The Alternating Least Squares (ALS) model was initialy implemented using the `implicit` library, and later changed to `pyspark.ml.recommendation` for better scalability. This library enables hyperparameter tuning and model evaluation using cross-validation.

FOr more detail, you can check the `ALS.ipynb` notebook.

### Model 2: Two-Tower Neural Network
The Two-Tower Neural Network model was more promising than the ALS model, and easier to use. It was implemented using the Keras library. The model uses a subset of `big_matrix.csv` for training and a subset of `small_matrix.csv` for evaluation.

The model focused mostly on using video features, and PCA was used to reduce the dimensionality as it was too much for my graphic card.

For more detail, you can check the `Two_Tower.ipynb` notebook.

### Model 3: Cosine Similarity
This model is a naive test to see if the cosine similarity between the video features is a good metric to use for recommendation.

For more detail, you can check the `cosine_similarity.ipynb` notebook.
