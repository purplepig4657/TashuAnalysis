import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import ClusterDataCN


class ClusterIndexColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        if ClusterDataCN.INDEX in X.columns:
            X.drop([ClusterDataCN.INDEX], axis=1, inplace=True)
        return X
