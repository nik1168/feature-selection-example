import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

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

        model = ExtraTreesClassifier()
        model.fit(X, y)
        print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.show()
