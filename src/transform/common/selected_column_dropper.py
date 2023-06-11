import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import TimeDataCN


class SelectedColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, selected_columns: list = None):
        self.__selected_columns = selected_columns

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        X.drop(self.__selected_columns, axis=1, inplace=True)
        return X
