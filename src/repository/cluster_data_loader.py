import os

import pandas as pd

from src.base.singleton_meta import SingletonMeta


import os

import pandas as pd

from src.base.singleton_meta import SingletonMeta


class ClusterDataLoader(metaclass=SingletonMeta):
    def __init__(self, cluster_data_information: dict = None):
        self.__FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.__DATA_DIRECTORY = os.path.join(self.__FILE_PATH, "data")

        self.__cluster_data_information = {
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "random": "random"
        } if cluster_data_information is None else cluster_data_information

        self.__all_cluster_data = self.__load_all_data()

    def __load_data(self, name: str) -> pd.DataFrame:
        file_name = f"station_clustering_{name}.csv"
        csv_file = os.path.join(self.__DATA_DIRECTORY, file_name)
        data = pd.read_csv(csv_file)
        return data

    def __load_all_data(self) -> dict:
        result = dict()
        for key, value in self.__cluster_data_information.items():
            result[key] = self.__load_data(value)
        return result

    def get_specific_data(self, name: str) -> pd.DataFrame:
        if name not in self.__all_cluster_data.keys():
            raise NameError(f"Name {name} is not in all_rent_data.")
        return self.all_data[name]

    @property
    def available_data_list(self) -> list:
        return list(self.__cluster_data_information.keys())

    @property
    def all_data(self) -> dict:
        return self.__all_cluster_data
