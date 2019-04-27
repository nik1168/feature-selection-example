from FileManager import FileManager
from UnivariateSelection import UnivariateSelection

if __name__ == "__main__":
    print("Here is where the magic of feature selection begins :)")
    file_manager = FileManager()
    data_set_ = file_manager.get_data_train()
    univariate_selection = UnivariateSelection(data_set_)
    univariate_selection.exec()


