from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import WeatherDataCN


class WeatherDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Weather data missed value preprocessing.
    """

    def __init__(self):
        pass

    def fit(self, X: dict, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = self.precipitation_missed_value_fill_zero(X)
        X = self.sunshine_duration_missed_value_fill_zero(X)
        X = self.weather_number_missed_value_fill_zero(X)
        X = self.other_missed_value_fill_average(X)
        return X

    # noinspection PyMethodMayBeStatic
    def precipitation_missed_value_fill_zero(self, X: pd.DataFrame) -> pd.DataFrame:
        X[WeatherDataCN.PRECIPITATION].fillna(0, inplace=True)
        return X

    # noinspection PyMethodMayBeStatic
    def sunshine_duration_missed_value_fill_zero(self, X: pd.DataFrame) -> pd.DataFrame:
        X[WeatherDataCN.SUNSHINE_DURATION].fillna(0, inplace=True)
        return X

    # noinspection PyMethodMayBeStatic
    def weather_number_missed_value_fill_zero(self, X: pd.DataFrame) -> pd.DataFrame:
        X[WeatherDataCN.WEATHER_NUMBER].fillna(0, inplace=True)
        return X

    # noinspection PyMethodMayBeStatic
    def other_missed_value_fill_average(self, X: pd.DataFrame) -> pd.DataFrame:
        X.fillna(X.mean(numeric_only=True), inplace=True)
        return X
