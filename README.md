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

#### Description
ALS is a collaborative filtering algorithm. It works by factorizing the user-item interaction matrix into two lower-dimensional matrices: one for users and one for items. The goal is to minimize the difference between the original interaction matrix and the product of the two lower-dimensional matrices.

Considering that we only have implicit feedback, ALS can work well. We will not use demographic data for this simple model. This algorithm is mostly used for sparse datasets.

#### Evaluation
Evaluation is a used feedback for the model training. We will use the `RegressionEvaluator` class from `pyspark.ml.evaluation` to evaluate the model with the rmse metric.

This model uses almost no feature to none, and is pretty basic. For evaluation, `kuairec_caption_category.csv` was used to get the caption of the recommended videos.

#### Conclusion
I encountered problems training the ALS model with the `big_matrix.csv` dataset. So I used the dataset `small_matrix.csv` instead.

The ALS model served as a baseline collaborative filtering method for the KuaiRec short video recommendation task. However, its performance was limited by several factors:

- **Data Constraints:** Due to computational issues, only the dense but small small_matrix.csv could be used for training, restricting the model’s ability to learn from the full diversity and sparsity of the dataset.

- **Model Simplicity:** The model did not leverage any side information or advanced features, relying solely on user-item interaction data.

Overall, ALS is a good starting point and a classic collaborative filtering approach, but its performance is limited by the computing resources.

#### Possible extensions
Given the limitations of ALS and the results observed, several possible extensions could be considered to improve performance:

- **More features:** Adding more video features (kuareic_caption_category) and user features (social_network).

- **Switch to Larger, Sparser Matrices:** Address the scalability and memory issues that prevented the use of big_matrix.csv. This could involve distributed training, more efficient data pipelines, or dimensionality reduction before factorization.

- **Advanced Matrix Factorization Techniques:** Explore more sophisticated matrix factorization approaches, such as Bayesian Personalized Ranking (BPR) or Weighted Matrix Factorization, which are often better suited for implicit feedback and large-scale data.

### Model 2: Two-Tower Neural Network
The Two-Tower Neural Network model was more promising than the ALS model, and easier to use. It was implemented using the Keras library. The model uses the `big_matrix.csv` dataset for training and evaluation.

The model focused mostly on using video features, and PCA was used to reduce the dimensionality as it was too much for my graphic card.

For more detail, you can check the `Two_Tower.ipynb` notebook.

### Model 3: Cosine Similarity
This model is a naive test to see if the cosine similarity between the video features is a good metric to use for recommendation.

#### Description
Cosine similarity is a metric used to measure how similar two vectors are. Users and items can be represented as vectors in a multi-dimensional space, and cosine similarity can be used to find the most similar items to a given item or the most similar users to a given user.

Unlike collaborative filtering approaches that rely on user-item interaction patterns, this method leverages explicit item and user features, making it suitable even when interaction data is sparse.

#### Evaluation
The system is evaluated as a binary classification problem. Given a set of recommended items, we assess whether the user genuinely liked the item. This is determined by checking if the user watched more than a defined threshold percentage (e.g., 70%) of the video.

For each recommendation, the label is:
- **Positive (1):** if the user watched more than the threshold (for example, watch_ratio > 0.7)
- **Negative (0):** otherwise

#### Conclusion
This model is very simple, yet its performance is highly sensitive to data sparsity.

On the denser `small_matrix`, the results are generally better because it's easier to find meaningful user similarities and verify user-item interactions. However, on the sparser matrix `big_matrix`, performance drops significantly. This decline isn't necessarily due to poor recommendations, but rather to limitations in the evaluation process itself: since we can only validate a recommendation if the user actually interacted with the item in the test set, many relevant suggestions go unrecognized.

As a result, the model may appear to perform worse than it truly does. 

One way to solve this problem is to use `big_matrix` for the "training" and `small_matrix` for label verification.


#### Possible extensions
This model can be improved with:
- **More features:** Adding more video features (kuareic_caption_category) and user features (social_network).
- **History-based recommendations:** Prioritizing items similar to the user's watch history.
- **Friend weighting:** Weighting the recommendations based on friendship connections.
- **Popularity scoring:** Incorporating item popularity into the item feature vector. More popular items may have a higher chance of being watched.