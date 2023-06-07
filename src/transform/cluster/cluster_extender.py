import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, WeatherDataCN, TimeDataCN, ClusterDataCN


class ClusterExtender(BaseEstimator, TransformerMixin):
    """
    Extend weather data column.
    """

    def __init__(self, cluster_data: pd.DataFrame):
        if cluster_data is None:
            raise "Required data is none."

        self.__cluster_data = cluster_data

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return self.extend_cluster_data(X)

    def extend_cluster_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self.__cluster_data.rename(columns={
            ClusterDataCN.STATION: RentDataCN.RENT_STATION
        }, inplace=True)

        data = pd.merge(data, self.__cluster_data, on=RentDataCN.RENT_STATION)
        data.drop([RentDataCN.RENT_STATION], axis=1, inplace=True)

        return data
