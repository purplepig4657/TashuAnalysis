import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, ClusterDataCN
from src.repository.cluster_data_loader import ClusterDataLoader


class ClusterColumnExtender(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.__data_loader = ClusterDataLoader()
        self.__cluster_data = self.__data_loader.data
        self.__cluster_data = self.__cluster_data[[
            ClusterDataCN.RENT_STATION,
            ClusterDataCN.CLUSTER
        ]]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return self.extend_cluster_data(X)

    def extend_cluster_data(self, X: pd.DataFrame):
        joined_X = pd.merge(X, self.__cluster_data, on=RentDataCN.RENT_STATION)
        return joined_X
