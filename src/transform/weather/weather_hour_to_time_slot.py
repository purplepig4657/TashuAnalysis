from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import TimeDataCN, TimeDataValue


class WeatherHourToTimeSlot(BaseEstimator, TransformerMixin):
    """
    Date columns to time slot column.
    """

    def __init__(self):
        pass

    def fit(self, X: dict, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.hour_to_time_slot(X)

    def hour_to_time_slot(self, X: pd.DataFrame) -> pd.DataFrame:
        X[TimeDataCN.TIME_CATEGORY] = X[TimeDataCN.HOUR].apply(self.classify_time)
        X.drop([TimeDataCN.HOUR], axis=1, inplace=True)
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
