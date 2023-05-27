import os

import pandas as pd

from src.base.singleton_meta import SingletonMeta


class DataLoader(metaclass=SingletonMeta):
    def __init__(self, rent_data_information: dict = None, station_data_information: dict = None):
        self.__FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.__DATA_DIRECTORY = os.path.join(self.__FILE_PATH, "data")
        self.__STATION_DATA = "location_of_public_bicycle_in_daejeon"

        self.__rent_data_information = {
            "2022_2": (2022, 2, None),
            "2022_1": (2022, 1, None),
            "2021": (2021, None, None),
            "2020": (2020, None, 'cp949'),
            "2019": (2019, None, None),
            "2018": (2018, None, None),
            "2017": (2017, None, None),
            "2016": (2016, None, 'cp949')
        } if rent_data_information is None else rent_data_information

        self.__station_data_information = {
            "2023": (2023, None),
            "2021": (2021, None),
            "2020": (2020, 'cp949')
        } if station_data_information is None else station_data_information

        self.__all_rent_data = self.__load_all_rent_data()
        self.__all_station_data = self.__load_all_station_data()

    def __load_rent_data(self, year: int, season: int = None, encoding: str = None) -> pd.DataFrame:
        file_name = (f"{year}_{season}" if season is not None else str(year)) + ".csv"
        csv_file = os.path.join(self.__DATA_DIRECTORY, file_name)
        data = pd.read_csv(csv_file, encoding=encoding)
        return data

    def __load_station_data(self, year: int, encoding: str = None) -> pd.DataFrame:
        file_name = f"{self.__STATION_DATA}_{year}.csv"
        csv_file = os.path.join(self.__DATA_DIRECTORY, file_name)
        data = pd.read_csv(csv_file, encoding=encoding)
        return data

    def __load_all_rent_data(self) -> dict:
        result = dict()
        for key, value in self.__rent_data_information.items():
            result[key] = self.__load_rent_data(*value)
        return result

    def __load_all_station_data(self) -> dict:
        result = dict()
        for key, value in self.__station_data_information.items():
            result[key] = self.__load_station_data(*value)
        return result

    def get_specific_rent_data(self, name: str) -> pd.DataFrame:
        if name not in self.__all_rent_data.keys():
            raise NameError(f"Name {name} is not in all_rent_data.")
        return self.all_rent_data[name]

    def get_specific_station_data(self, name: str) -> pd.DataFrame:
        if name not in self.__all_station_data.keys():
            raise NameError(f"Name {name} is not in all_station_data.")
        return self.all_station_data[name]

    @property
    def available_rent_data_list(self) -> list:
        return list(self.__rent_data_information.keys())

    @property
    def available_station_data_list(self) -> list:
        return list(self.__station_data_information.keys())

    @property
    def all_rent_data(self) -> dict:
        return self.__all_rent_data

    @property
    def all_station_data(self) -> dict:
        return self.__all_station_data
