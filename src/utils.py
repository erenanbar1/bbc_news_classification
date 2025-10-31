import pandas as pd
import numpy as np

def confusion_matrix(comparison_matrix):
    actual = comparison_matrix['category']
    predicted = comparison_matrix['predicted_category']
    labels = np.sort(np.unique(np.concatenate([actual.values, predicted.values])))
    cm = pd.DataFrame(
        0, index=labels, columns=labels, dtype=int
    )
    for a, p in zip(actual, predicted):
        cm.loc[a, p] += 1
    return cm
