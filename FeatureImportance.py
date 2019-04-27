import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

"""
Feature importance gives you a score for each feature of your data, 
the higher the score more important or relevant is the feature towards your output variable.

Feature importance is an inbuilt class that comes with Tree Based Classifiers, 
we will be using Extra Tree Classifier for extracting the top 10 features for the dataset.
 """


class FeatureImportance:

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
