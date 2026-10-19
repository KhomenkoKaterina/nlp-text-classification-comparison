# Sentiment Analysis of Restaurant Reviews

This project is dedicated to the task of sentiment analysis (review classification) using various text vectorization methods and machine learning algorithms.

## Data Description

The dataset used is the Restaurant Reviews from Kaggle.

* **Total size**: 1,000 reviews.


* **Class distribution**: The dataset is perfectly balanced, containing 500 positive (`1`) and 500 negative (`0`) reviews.


* **Data split**: The data is divided into a training set of 750 samples and a testing set of 250 samples using a 75/25 split (`test_size=0.25`).



## Methods and Approaches

The primary classification model used across all text representations is **Logistic Regression**. To achieve the best accuracy and prevent overfitting, `GridSearchCV` was utilized for hyperparameter tuning (finding the optimal regularization parameter `C`) alongside cross-validation.

The following vectorization techniques were implemented and evaluated:

1. **Bag-of-Words (BoW)**:


* Implemented using `CountVectorizer`.


* The model was tested both with and without English stop words.


* The optimal regularization parameter identified by grid search was `C=5`.




2. **TF-IDF + n-grams**:


* Implemented using `TfidfVectorizer`.


* The pipeline evaluated different n-gram ranges: `(1, 1)`, `(1, 2)`, and `(1, 3)`.


* The most effective parameters found were `C=1` and unigrams `ngram_range=(1, 1)`.


* IDF metrics were analyzed to identify the most and least informative terms in the corpus.




3. **Word2Vec**:


* Custom tokenization was performed using `nltk.tokenize.word_tokenize`, keeping only alphabetic tokens.


* Dense vector embeddings were trained using `gensim.models.Word2Vec` with `vector_size=100` and `window=5`.


* Each review was represented as the average vector of its constituent words.





## Model Comparison Results

Based on the testing results, the models demonstrated the following accuracies:

* **BoW (baseline)**: `0.82` 


* **TF-IDF + n-grams**: `0.77`

* **Word2Vec**: `0.61`


The basic Bag-of-Words approach proved to be the most effective for this dataset.

## Tech Stack

* **Python**

* **Data Manipulation**: Pandas, NumPy


* **Machine Learning**: Scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`, `GridSearchCV`)


* **NLP**: NLTK (tokenization), Gensim (Word2Vec)
