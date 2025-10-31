import pandas as pd
import numpy as np

def train_parameters(X_train, Y_train, dirichlet_prior=1):
    X_train.loc[:] = (X_train.loc[:] > 0).astype(int) # Binarize features
    XY_train = pd.concat([X_train, Y_train], axis=1)

    class_counts = XY_train['category'].value_counts().sort_index()
    word_counts = XY_train.groupby('category').sum(numeric_only=True).sort_index()

    # smoothing
    theta = (word_counts + dirichlet_prior).div(class_counts.values[:, None] + 2 * dirichlet_prior)

    # class priors
    prior_probs = class_counts / len(XY_train)
    log_priors = np.log(prior_probs)

    word_probs = np.log(theta)

    return word_probs, log_priors


def predict(X_test, log_priors, log_word_probs):
    X = X_test.reindex(columns=log_word_probs.columns, fill_value=0)
    X = (X > 0).astype(int)

    log_p1 = log_word_probs
    log_p0 = np.log1p(-np.exp(log_p1).clip(upper=1 - 1e-12))

    scores = X.dot(log_p1.T) + (1 - X).dot(log_p0.T)
    scores = scores.add(log_priors, axis=1)
    scores = scores.reindex(columns=sorted(scores.columns))
    preds = scores.idxmax(axis=1)

    return preds
