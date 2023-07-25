import os

import pandas as pd

from src.base.singleton_meta import SingletonMeta


class HolidayDataLoader(metaclass=SingletonMeta):
    def __init__(self, holiday_data_information: dict = None):
        self.__FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.__DATA_DIRECTORY = os.path.join(self.__FILE_PATH, "data")

        self.__holiday_data_information = {
            "basic": "holiday_info"
        } if holiday_data_information is None else holiday_data_information

        self.__all_holiday_data = self.__load_all_data()

    def __load_data(self, name: str) -> pd.DataFrame:
        file_name = f"{name}.csv"
        csv_file = os.path.join(self.__DATA_DIRECTORY, file_name)
        data = pd.read_csv(csv_file)
        return data

    def __load_all_data(self) -> dict:
        result = dict()
        for key, value in self.__holiday_data_information.items():
            result[key] = self.__load_data(value)
        return result

    def get_specific_data(self, name: str) -> pd.DataFrame:
        if name not in self.__all_holiday_data.keys():
            raise NameError(f"Name {name} is not in all_rent_data.")
        return self.all_data[name]

    @property
    def available_data_list(self) -> list:
        return list(self.__holiday_data_information.keys())

    @property
    def all_data(self) -> dict:
        return self.__all_holiday_data
