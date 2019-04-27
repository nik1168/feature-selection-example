import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Correlation states how the features are related to each other or the target variable

Correlation can be positive (increase in one value of feature increases the value of the target variable) 
or negative (increase in one value of feature decreases the value of the target variable)
 """


class Correlation:

    def __init__(self, data_set):
        self.data_set = data_set

    def exec(self):
        X = self.data_set.iloc[:, 0:20]  # independent columns
        y = self.data_set.iloc[:, -1]  # target column i.e price range
        # get correlations of each features in dataset
        corrmat = self.data_set.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(20, 20))
        # plot heat map
        g = sns.heatmap(self.data_set[top_corr_features].corr(), annot=True, cmap="RdYlGn")
        plt.show()
