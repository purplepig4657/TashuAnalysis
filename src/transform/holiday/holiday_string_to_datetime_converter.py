import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import HolidayDataCN


class HolidayStringToDatetimeConverter(BaseEstimator, TransformerMixin):
    """
    String date format to datetime type.
    """

    def __init__(self):
        self.__columns = [
            HolidayDataCN.DATE
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return self.str_to_datetime(X)

    def str_to_datetime(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in self.__columns:
            data[column] = pd.to_datetime(data[column].apply(str), format='%Y-%m-%d', errors='coerce')
        return data
