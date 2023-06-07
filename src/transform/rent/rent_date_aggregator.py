import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, TimeDataCN


class RentDateAggregator(BaseEstimator, TransformerMixin):
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

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        sampled_X = X[[
            RentDataCN.RENT_STATION,
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            self.__hour_column_name,
            TimeDataCN.WEEKDAY
        ]].copy()

        return self.aggregate(sampled_X)

    # noinspection PyMethodMayBeStatic
    def aggregate(self, X: pd.DataFrame) -> pd.DataFrame:
        X[RentDataCN.RENT_COUNT] = X.groupby([
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            self.__hour_column_name,
            RentDataCN.RENT_STATION
        ])[RentDataCN.RENT_STATION].transform('count')

        X.drop_duplicates(subset=[
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            self.__hour_column_name,
            RentDataCN.RENT_STATION
        ], inplace=True)

        return X
