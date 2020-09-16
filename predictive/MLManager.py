import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from data.DataManager import DataManager, ACCELEROMETER_PSEUDONAME


class MLManager:

    @staticmethod
    def logistic_regression(data):

        X_train, X_test, y_train, y_test = MLManager.split_train_test(data)

        if MLManager._has_only_one_class(y_train):
            return None, None, None, None, None

        model = LogisticRegression(solver='newton-cg')
        model.fit(X_train, y_train)

        return model, X_train, X_test, y_train, y_test

    @staticmethod
    def compate_output(outs_1, outs_2):
        goods = 0

        for out_1, out_2 in zip(outs_1, outs_2):
            if out_1 == out_2:
                goods += 1

        return goods / len(outs_1)

    @staticmethod
    def split_train_test(data, x=['msm_gyro'], y='PROBABLE', test_size=0.1):
        return train_test_split(data[x], data[y], test_size=test_size)

    @staticmethod
    def _has_only_one_class(y):
        return len(np.unique(y)) == 1
