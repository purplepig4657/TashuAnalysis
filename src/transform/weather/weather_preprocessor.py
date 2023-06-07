from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import WeatherDataCN


class WeatherPreprocessor(BaseEstimator, TransformerMixin):
    """
    Weather data missed value preprocessing.
    """

    def __init__(self):
        pass

    def fit(self, X: dict, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = self.precipitation_preprocessing(X)
        X = self.sunshine_duration_preprocessing(X)
        X = self.weather_number_preprocessing(X)
        X = self.other_missed_value_preprocessing(X)
        return X

    # noinspection PyMethodMayBeStatic
    def precipitation_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        X[WeatherDataCN.PRECIPITATION].fillna(0, inplace=True)
        return X

    # noinspection PyMethodMayBeStatic
    def sunshine_duration_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        X[WeatherDataCN.SUNSHINE_DURATION].fillna(0, inplace=True)
        return X

    # noinspection PyMethodMayBeStatic
    def weather_number_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        X[WeatherDataCN.WEATHER_NUMBER].fillna(0, inplace=True)
        return X

    # noinspection PyMethodMayBeStatic
    def other_missed_value_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        X.fillna(X.mean(numeric_only=True), inplace=True)
        return X
