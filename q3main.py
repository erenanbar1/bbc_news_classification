import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.multinomial_naive_bayes import train_parameters as train_multinom_naive_bayes, predict as predict_multinom_naive_bayes
from src.bernoulli_naive_bayes import train_parameters as train_bernoulli_naive_bayes, predict as predict_bernoulli_naive_bayes
from src.utils import confusion_matrix as compute_confusion_matrix

X_train = pd.read_csv("data/X_train.csv", sep=r"\s+")
Y_train = pd.read_csv("data/Y_train.csv", header=None, names=['category'])
X_test  = pd.read_csv("data/X_test.csv",  sep=r"\s+")
Y_test  = pd.read_csv("data/Y_test.csv",  header=None, names=['category'])

# Train Multinomial Naive Bayes model and make predictions for Q2.2
XY_train = pd.concat([X_train, Y_train], axis=1) # labeled data
param_matrix, log_priors = train_multinom_naive_bayes(XY_train) # kxn parameter matrix. k: num classes, n: num features
predictions = predict_multinom_naive_bayes(X_test, param_matrix, log_priors)
comparison_matrix = pd.concat([Y_test, pd.DataFrame(predictions, columns=['predicted_category'])], axis=1)
accuracy = np.mean(predictions == Y_test['category'].values)
confusion_matrix = compute_confusion_matrix(comparison_matrix)

print("********************************** Q2.2 Multinomial Naive Bayes Results **********************************")
print(f"Q2.2 Multinomial Naive Bayes Accuracy: {accuracy:.3f}")
print("Q2.2 Confusion Matrix:\n", confusion_matrix)
print("*******************************************************************************************************")

#train multinomial naive bayes with dirichlet prior smoothing alpha=1
param_matrix_smooth, log_priors_smooth = train_multinom_naive_bayes(XY_train, dirichlet_prior=1) # kxn parameter matrix. k: num classes, n: num features
predictions = predict_multinom_naive_bayes(X_test, param_matrix_smooth, log_priors_smooth)
comparison_matrix = pd.concat([Y_test, pd.DataFrame(predictions, columns=['predicted_category'])], axis=1)
accuracy = np.mean(predictions == Y_test['category'].values)
confusion_matrix = compute_confusion_matrix(comparison_matrix)

print("********************************** Q2.3 Multinomial Naive Bayes Results (dirichlet prior = 1)) **********************************")
print(f"Q2.3 Multinomial Naive Bayes Accuracy (smoothing alpha=1): {accuracy:.3f}")
print("Q2.3 Confusion Matrix :\n", confusion_matrix)
print("*******************************************************************************************************")


param_matrix_bernoulli, log_priors_bernoulli = train_bernoulli_naive_bayes(X_train, Y_train, dirichlet_prior=1)
predictions_bernoulli = predict_bernoulli_naive_bayes(X_test, log_priors_bernoulli, param_matrix_bernoulli)
comparison_matrix_bernoulli = pd.concat([Y_test, pd.DataFrame(predictions_bernoulli, columns=['predicted_category'])], axis=1)
accuracy_bernoulli = np.mean(predictions_bernoulli == Y_test['category'].values)
confusion_matrix_bernoulli = compute_confusion_matrix(comparison_matrix_bernoulli)

print("********************************** Q2.4 Bernoulli Naive Bayes Results (dirichlet prior = 1) **********************************")
print(f"Q2.4 Bernoulli Naive Bayes Accuracy (smoothing alpha=1): {accuracy_bernoulli:.3f}")
print("Q2.4 Confusion Matrix :\n", confusion_matrix_bernoulli)
print("*******************************************************************************************************")


