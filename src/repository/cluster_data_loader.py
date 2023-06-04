import os

import pandas as pd

from src.base.singleton_meta import SingletonMeta


class ClusterDataLoader(metaclass=SingletonMeta):
    def __init__(self):
        self.__FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.__DATA_DIRECTORY = os.path.join(self.__FILE_PATH, "data")

        self.data = self.__load_data()

    def __load_data(self) -> pd.DataFrame:
        file_name = "Station_Clustering_by_Time.csv"
        csv_file = os.path.join(self.__DATA_DIRECTORY, file_name)
        data = pd.read_csv(csv_file)
        return data
