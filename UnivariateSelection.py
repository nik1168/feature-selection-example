import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

"""
The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests
to select a specific number of features.

The example below uses the chi-squared (chi²) statistical test for non-negative features to select 10 of the best 
features from the Mobile Price Range
 """


class UnivariateSelection:

    def __init__(self, data_set):
        self.data_set = data_set

    def exec(self):
        X = self.data_set.iloc[:, 0:20]  # independent columns
        y = self.data_set.iloc[:, -1]  # target column i.e price range

        # apply SelectKBest class to extract top 10 best features
        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(X, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        # concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
        print(featureScores.nlargest(10, 'Score'))  # print 10 best features
