# BBC News Classification 

This project focuses on **classifying BBC News articles** into one of five categories using different variants of the **Naive Bayes** algorithm. The dataset is preprocessed, news article are represented as word frequencies. The documents are modeled both as bag-of-words and 

It demonstrates text preprocessing, feature extraction with the Bag-of-Words model, and comparative analysis of multiple probabilistic classifiers. 

---

## Project Overview

The BBC dataset consists of thousands of labeled news articles from five categories:

- **Business**
- **Entertainment**
- **Politics**
- **Sport**
- **Tech**

The goal is to train models that can accurately predict the category of unseen news articles.

---

## Models Implemented

Three Naive Bayes variants were trained and evaluated:

1. **Multinomial Naive Bayes**
   - Uses raw word frequencies.
   - Performs well when word count information matters.
   - Sensitive to zero-frequency problems for unseen words.

2. **Multinomial Naive Bayes with Additive Smoothing**
   - Applies Laplace smoothing (α = 1).
   - Handles unseen words more gracefully and improves accuracy stability.

3. **Bernoulli Naive Bayes**
   - Binary features: word presence (1) or absence (0).
   - Performs better on short, keyword-rich texts or when exact word counts are less informative.

Each model’s **testing accuracy** and **confusion matrix** were compared to analyze their performance differences.

---





