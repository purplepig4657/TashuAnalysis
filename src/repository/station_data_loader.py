import os

import pandas as pd

from src.base.singleton_meta import SingletonMeta


class StationDataLoader(metaclass=SingletonMeta):
    def __init__(self, station_data_information: dict = None):
        self.__FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.__DATA_DIRECTORY = os.path.join(self.__FILE_PATH, "data")
        self.__STATION_DATA = "location_of_public_bicycle_in_daejeon"

        self.__station_data_information = {
            "2023": (2023, None),
            "2021": (2021, None),
            "2020": (2020, 'cp949')
        } if station_data_information is None else station_data_information

        self.__all_station_data = self.__load_all_data()

    def __load_data(self, year: int, encoding: str = None) -> pd.DataFrame:
        file_name = f"{self.__STATION_DATA}_{year}.csv"
        csv_file = os.path.join(self.__DATA_DIRECTORY, file_name)
        data = pd.read_csv(csv_file, encoding=encoding)
        return data

    def __load_all_data(self) -> dict:
        result = dict()
        for key, value in self.__station_data_information.items():
            result[key] = self.__load_data(*value)
        return result

    def get_specific_data(self, name: str) -> pd.DataFrame:
        if name not in self.__all_station_data.keys():
            raise NameError(f"Name {name} is not in all_station_data.")
        return self.all_data[name]

    @property
    def available_data_list(self) -> list:
        return list(self.__station_data_information.keys())

    @property
    def all_data(self) -> dict:
        return self.__all_station_data
