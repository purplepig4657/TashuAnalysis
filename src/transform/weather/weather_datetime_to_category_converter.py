from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import WeatherDataCN, TimeDataCN


class WeatherDatetimeToCategoryConverter(BaseEstimator, TransformerMixin):
    """
    Datetime data to categorical data.
    """

    def __init__(self):
        self.__datetime_column = WeatherDataCN.DATE

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        extended_X = self.categorical_datetime_extender(X)
        extended_X.drop(self.__datetime_column, axis=1, inplace=True)
        return X

    def categorical_datetime_extender(self, X: pd.DataFrame) -> pd.DataFrame:
        X[TimeDataCN.YEAR] = X[self.__datetime_column].dt.year
        X[TimeDataCN.MONTH] = X[self.__datetime_column].dt.month
        X[TimeDataCN.DAY] = X[self.__datetime_column].dt.day
        X[TimeDataCN.HOUR] = X[self.__datetime_column].dt.hour
        X[TimeDataCN.WEEKDAY] = X[self.__datetime_column].dt.weekday

        return X
