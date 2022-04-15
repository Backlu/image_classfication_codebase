# coding: utf-8

#History
# 4/15: first imp ver.

from numpy import dstack
import joblib
from sklearn.linear_model import LogisticRegression

def stacked_dataset(pred_list):
    stackX = None
    for yhat in pred_list:
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

def fit_stacked_model(stackedX, inputy):
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model
