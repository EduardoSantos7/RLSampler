import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from data.DataManager import DataManager, ACCELEROMETER_PSEUDONAME


class MLManager:

    @staticmethod
    def logistic_regression(data):

        X_train, X_test, y_train, y_test = train_test_split(
            data[['msm_gyro']], data.PROBABLE, test_size=0.1)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        return model, X_train, X_test, y_train, y_test

    @staticmethod
    def compate_outpt(outs_1, outs_2):
        goods = 0

        for out_1, out_2 in zip(outs_1, outs_2):
            if out_1 == out_2:
                goods += 1

        return goods / len(outs_1)
