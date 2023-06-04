import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, TimeDataCN


class SpecificTimeSlotAggregator(BaseEstimator, TransformerMixin):
    """
    Simple Aggregate data per datetime hour and per rent_station.
    depends_on: :py:class:`src.transform.transformer.column_renamer.ColumnRenamer`,
                :py:class:`src.transform.transformer.string_to_datetime_converter.StringToDatetimeConverter`
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        sampled_X = X[[
            RentDataCN.RENT_STATION,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.HOUR,
            TimeDataCN.WEEKDAY
        ]].copy()
        return self.aggregate(sampled_X)

    # noinspection PyMethodMayBeStatic
    def aggregate(self, X: pd.DataFrame) -> pd.DataFrame:
        X[TimeDataCN.TIME_CATEGORY] = X[TimeDataCN.HOUR].apply(self.classify_time)
        X[RentDataCN.RENT_COUNT] = X.groupby([
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.TIME_CATEGORY,
            RentDataCN.RENT_STATION
        ])[RentDataCN.RENT_STATION].transform('count')
        X.drop_duplicates(subset=[
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.TIME_CATEGORY,
            RentDataCN.RENT_STATION
        ], inplace=True)
        X.drop_duplicates(subset=[
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.TIME_CATEGORY,
            RentDataCN.RENT_STATION
        ], inplace=True)
        return X

    # noinspection PyMethodMayBeStatic
    def classify_time(self, hour):
        if 7 <= hour <= 9:
            return TimeDataCN.MORNING_PEAK
        elif 10 <= hour <= 12:
            return TimeDataCN.MORNING_NON_PEAK
        elif 13 <= hour <= 16:
            return TimeDataCN.AFTERNOON_NON_PEAK
        elif 17 <= hour <= 19:
            return TimeDataCN.AFTERNOON_PEAK
        else:
            return TimeDataCN.NIGHTTIME
