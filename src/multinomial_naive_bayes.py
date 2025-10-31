import pandas as pd
import numpy as np
def train_parameters(XY_train, dirichlet_prior=0):
    word_freq_sum = XY_train.groupby('category').sum()
    # apply dirichlet prior smoothing
    word_freq_sum += dirichlet_prior
    total_words_per_category = word_freq_sum.sum(axis=1)
    word_probs = word_freq_sum.div(total_words_per_category, axis=0)

    # take log normally (this will give -inf where prob=0)
    log_param_matrix = np.log(word_probs)
    log_param_matrix.fillna(-np.inf, inplace=True)

    # compute log priors
    priors = XY_train['category'].value_counts().sort_index() / len(XY_train)
    log_priors = np.log(priors)
    log_priors.fillna(-np.inf, inplace=True)

    return log_param_matrix, log_priors


def predict(X_test, log_param_matrix, log_priors):
    # align test vocab to training vocab
    X = X_test.reindex(columns=log_param_matrix.columns, fill_value=0)
    
    # matrix multiplication (pandas takes care of alignment)
    scores = X.dot(log_param_matrix.T)
    
    # add priors columnwise
    scores = scores.add(log_priors, axis=1)
    scores.fillna(-np.inf, inplace=True)
    # take argmax by column
    preds = scores.idxmax(axis=1)
    return preds