import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, TimeDataCN


class SimpleDatetimeAggregator(BaseEstimator, TransformerMixin):
    """
    Simple Aggregate data per datetime hour and per rent_station.
    depends_on: :py:class:`src.transform.transformer.column_renamer.ColumnRenamer`,
                :py:class:`src.transform.transformer.string_to_datetime_converter.StringToDatetimeConverter`
    """

    def __init__(self, is_categorical: bool = False):
        self.__is_categorical = is_categorical

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        if self.__is_categorical:
            sampled_X = X[[
                RentDataCN.RENT_STATION,
                TimeDataCN.MONTH,
                TimeDataCN.DAY,
                TimeDataCN.HOUR,
                TimeDataCN.WEEKDAY
            ]].copy()
            return self.aggregate_by_date_category(sampled_X)
        else:
            sampled_X = X[[
                RentDataCN.RENT_STATION,
                RentDataCN.RENT_DATE,
            ]].copy()
            return self.aggregate_by_datetime(sampled_X)

    # noinspection PyMethodMayBeStatic
    def aggregate_by_datetime(self, X: pd.DataFrame) -> pd.DataFrame:
        X[RentDataCN.RENT_DATE] = X[RentDataCN.RENT_DATE].dt.floor('H')
        X[RentDataCN.RENT_COUNT] = X.groupby([
            RentDataCN.RENT_DATE,
            RentDataCN.RENT_STATION
        ])[RentDataCN.RENT_STATION].transform('count')
        X.drop_duplicates(subset=[RentDataCN.RENT_DATE, RentDataCN.RENT_STATION], inplace=True)
        return X

    # noinspection PyMethodMayBeStatic
    def aggregate_by_date_category(self, X: pd.DataFrame) -> pd.DataFrame:
        X[RentDataCN.RENT_COUNT] = X.groupby([
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.HOUR,
            RentDataCN.RENT_STATION
        ])[RentDataCN.RENT_STATION].transform('count')
        X.drop_duplicates(subset=[
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            TimeDataCN.HOUR,
            RentDataCN.RENT_STATION
        ], inplace=True)
        return X
