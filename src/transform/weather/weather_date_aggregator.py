import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import WeatherDataCN, WeatherDataValue, TimeDataCN


class WeatherDateAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregate data with specific time slot
    """

    def __init__(self, is_categorized: bool = False):
        if is_categorized:
            self.__hour_column_name = TimeDataCN.TIME_CATEGORY
        else:
            self.__hour_column_name = TimeDataCN.HOUR

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        sampled_X = X[[
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            self.__hour_column_name,
            TimeDataCN.WEEKDAY,
            WeatherDataCN.TEMPERATURE,
            WeatherDataCN.PRECIPITATION,
            WeatherDataCN.SUNSHINE_DURATION
        ]].copy()
        return self.aggregate(sampled_X)

    def aggregate(self, X: pd.DataFrame) -> pd.DataFrame:
        X[WeatherDataCN.RAINFALL] = X.groupby([
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            self.__hour_column_name,
            WeatherDataCN.PRECIPITATION
        ])[WeatherDataCN.PRECIPITATION].transform(
            lambda x: WeatherDataValue.RAIN if any(x > 0.5) else WeatherDataValue.NON_RAIN)

        X[WeatherDataCN.TEMPERATURE] = X.groupby([
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            self.__hour_column_name,
            WeatherDataCN.TEMPERATURE
        ])[WeatherDataCN.TEMPERATURE].transform('mean')

        X.drop_duplicates(subset=[
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            self.__hour_column_name
        ], inplace=True)

        X.drop([WeatherDataCN.PRECIPITATION], axis=1, inplace=True)

        return X
