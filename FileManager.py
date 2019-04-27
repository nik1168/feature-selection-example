import pandas as pd

from constants import DATA_TRAIN, DATA_TEST


class FileManager:
    def __init__(self):
        pass

    def get_data_train(self):
        return pd.read_csv(DATA_TRAIN)

    def get_data_test(self):
        return pd.read_csv(DATA_TEST)
