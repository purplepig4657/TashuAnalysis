import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, WeatherDataCN


class StringToDatetimeConverter(BaseEstimator, TransformerMixin):
    """
    String date format to datetime type.
    depends_on: :py:class:`src.transform.transformer.column_renamer.ColumnRenamer`
    """

    def __init__(self, data_category: str = 'rent', per_hour: bool = False):
        self.__data_category = data_category
        self.__per_hour = per_hour
        if data_category == 'rent':
            self.__columns = [
                RentDataCN.RENT_DATE,
                RentDataCN.RETURN_DATE
            ]
        elif data_category == 'weather':
            self.__columns = [
                WeatherDataCN.DATE
            ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.__data_category == 'rent':
            return self.rent_str_to_datetime(X)
        elif self.__data_category == 'weather':
            return self.weather_str_to_datetime(X)

    def rent_str_to_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in self.__columns:
            data.dropna(subset=[column], axis=0, inplace=True)
            data[column] = data[column].astype('int')
            data[column] = pd.to_datetime(data[column].apply(str), format='%Y%m%d%H%M%S', errors='coerce')
        for column in self.__columns:
            data[column] = data[column].dt.floor('H')
        return data

    def weather_str_to_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in self.__columns:
            data[column] = pd.to_datetime(data[column].apply(str), format='%Y-%m-%d %H:%M:%S', errors='coerce')
        for column in self.__columns:
            data[column] = data[column].dt.floor('H')
        return data
