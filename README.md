# Sentiment Analysis of Restaurant Reviews

This project is dedicated to the task of sentiment analysis (review classification) using various text vectorization methods and machine learning.

## Data Description

The dataset used is the [Restaurant Reviews from Kaggle](https://www.kaggle.com/datasets/shub99/sentiment-analysis-data).

* **Size**: 1,000 reviews (500 positive and 500 negative).
* **Target variable (`Liked`)**: `1` — positive review, `0` — negative review.

## Methods and Approaches

During the study, the following approaches to text preprocessing and vector representation were implemented and tested in conjunction with logistic regression (`LogisticRegression`):

1. **Bag-of-Words (BoW)** — basic word frequency representation.
2. **TF-IDF + n-grams** — accounting for word importance using n-grams.
3. **Word2Vec** — dense vector word embeddings (`gensim` library).

`GridSearchCV` was used to find the optimal hyperparameters for the models.

## Model Comparison Results (Accuracy)

Based on the testing results, the models demonstrated the following accuracy:

* **BoW (baseline)**: `0.82` 
* **TF-IDF + n-grams**: `0.77`
* **Word2Vec**: `0.61`

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (CountVectorizer, TfidfVectorizer, LogisticRegression, GridSearchCV)
* NLTK
* Gensim (Word2Vec)
