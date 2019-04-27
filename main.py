from Correlation import Correlation
from FeatureImportance import FeatureImportance
from FileManager import FileManager
from UnivariateSelection import UnivariateSelection

if __name__ == "__main__":
    file_manager = FileManager()
    data_set = file_manager.get_data_train()
    print(data_set.columns)
    univariate_selection = UnivariateSelection(data_set)
    univariate_selection.exec()
    feature_importance = FeatureImportance(data_set)
    feature_importance.exec()
    correlation = Correlation(data_set)
    correlation.exec()


