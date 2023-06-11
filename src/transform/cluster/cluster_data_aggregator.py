import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, TimeDataCN, ClusterDataCN


class ClusterDataAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, is_categorized: bool = False):
        if is_categorized:
            self.__hour_column_name = TimeDataCN.TIME_CATEGORY
        else:
            self.__hour_column_name = TimeDataCN.HOUR

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self.aggregate(X)

    # noinspection PyMethodMayBeStatic
    def aggregate(self, X: pd.DataFrame) -> pd.DataFrame:
        X[RentDataCN.RENT_COUNT] = X.groupby([
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            self.__hour_column_name,
            ClusterDataCN.CLUSTER
        ])[ClusterDataCN.CLUSTER].transform('count')

        X.drop_duplicates(subset=[
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY,
            self.__hour_column_name,
            ClusterDataCN.CLUSTER
        ], inplace=True)

        return X
