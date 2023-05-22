import os

import pandas as pd


class DataLoader:
    def __init__(self, data_information: dict = None):
        self.__FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.__DATA_DIRECTORY = os.path.join(self.__FILE_PATH, "data")
        self.__STATION_DATA = "location_of_public_bicycle_in_daejeon_20200801"

        self.__data_information = {
            "2022_2": (2022, 2, None),
            "2022_1": (2022, 1, None),
            "2021": (2021, None, None),
            "2020": (2020, None, 'cp949'),
            "2019": (2019, None, None),
            "2018": (2018, None, None),
            "2017": (2017, None, None),
            "2016": (2016, None, 'cp949')
        } if data_information is None else data_information

        self.__all_data = self.load_all_data()
        self.__station_data = self.__load_station_data(encoding='cp949')

    def __load_data(self, year: int, season: int = None, encoding: str = None) -> pd.DataFrame:
        file_name = (f"{year}_{season}" if season is not None else str(year)) + ".csv"
        csv_file = os.path.join(self.__DATA_DIRECTORY, file_name)
        data = pd.read_csv(csv_file, encoding=encoding)
        return data

    def __load_station_data(self, encoding: str = None) -> pd.DataFrame:
        file_name = f"{self.__STATION_DATA}.csv"
        csv_file = os.path.join(self.__DATA_DIRECTORY, file_name)
        data = pd.read_csv(csv_file, encoding=encoding)
        return data

    def load_all_data(self) -> dict:
        result = dict()
        for key, value in self.__data_information.items():
            result[key] = self.__load_data(*value)
        return result

    def get_specific_data(self, name: str) -> pd.DataFrame:
        if name not in self.__all_data.keys():
            raise NameError(f"Name {name} is not in all_data.")
        return self.all_data[name]

    @property
    def available_data_list(self) -> list:
        return list(self.__data_information.keys())

    @property
    def all_data(self) -> dict:
        return self.__all_data

    @property
    def station_data(self) -> pd.DataFrame:
        return self.__station_data
