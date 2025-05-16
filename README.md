# Short Video Recommender System (KuaiRec)

This repository contains the implementation of different short video recommender systems with the KuaiRec dataset.

---

## **Table of Contents**

- [Setup](#setup)
- [Dataset](#dataset)
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

## Models
Multiple models were tested for this project. They are listed in chronological order of implementation.

### Model 1: ALS
The Alternating Least Squares (ALS) model was initialy implemented using the `implicit` library, and later changed to `pyspark.ml.recommendation` for better scalability. This library enables hyperparameter tuning and model evaluation using cross-validation.

I encountered problems training the ALS model with the `big_matrix.csv` dataset. So I used the dataset `small_matrix.csv` instead.

This model uses almost no feature to none, and is pretty basic. For evaluation, `kuairec_caption_category.csv` was used to get the caption of the recommended videos.

### Model 2: Two-Tower Neural Network
The Two-Tower Neural Network model was more promising than the ALS model, and easier to use. It was implemented using the Keras library. The model uses the `big_matrix.csv` dataset for training and evaluation.

The model focused mostly on using video features, and PCA was used to reduce the dimensionality as it was too much for my graphic card.

For more detail, you can check the `Two_Tower.ipynb` notebook.

### Model 3: Cosine Similarity
This model is a test to see if the cosine similarity between the video features is a good metric to use for recommendation.