import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, WeatherDataCN, WeatherDataValue, TimeDataCN, TimeDataValue


class TemperatureAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        sampled_X = X[[
            RentDataCN.RENT_STATION,
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.HOUR,
            TimeDataCN.WEEKDAY,
            WeatherDataCN.TEMPERATURE,
        ]].copy()
        return self.aggregate(sampled_X)

    # noinspection PyMethodMayBeStatic
    def aggregate(self, X: pd.DataFrame) -> pd.DataFrame:
        X[TimeDataCN.TIME_CATEGORY] = X[TimeDataCN.HOUR].apply(self.classify_time)
        X[RentDataCN.RENT_COUNT] = X.groupby([
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.TIME_CATEGORY,
            RentDataCN.RENT_STATION
        ])[RentDataCN.RENT_STATION].transform('count')

        X[WeatherDataCN.TEMPERATURE_AVG] = X.groupby([
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.TIME_CATEGORY,
            WeatherDataCN.TEMPERATURE
        ])[WeatherDataCN.TEMPERATURE].transform('mean')

        X.drop_duplicates(subset=[
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.TIME_CATEGORY,
            RentDataCN.RENT_STATION
        ], inplace=True)

        X.drop([WeatherDataCN.TEMPERATURE, TimeDataCN.YEAR], axis=1, inplace=True)

        return X

    # noinspection PyMethodMayBeStatic
    def classify_time(self, hour):
        if 7 <= hour <= 9:
            return TimeDataValue.MORNING_PEAK
        elif 10 <= hour <= 12:
            return TimeDataValue.MORNING_NON_PEAK
        elif 13 <= hour <= 16:
            return TimeDataValue.AFTERNOON_NON_PEAK
        elif 17 <= hour <= 19:
            return TimeDataValue.AFTERNOON_PEAK
        else:
            return TimeDataValue.NIGHTTIME
