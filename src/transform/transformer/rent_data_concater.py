from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class RentDataConcater(BaseEstimator, TransformerMixin):
    """
    Concatenating rent data.
    """

    def __init__(self, concat_data_list: list = None):
        if concat_data_list is None:
            concat_data_list = ["2016", "2017", "2018", "2019", "2020", "2021", "2022_1"]
        self.__data_list = concat_data_list

    def fit(self, X: dict, y=None):
        return self

    def transform(self, X: dict, y=None):
        return self.concatenate(X)

    def concatenate(self, X: dict):
        data_list = [value for key, value in X.items() if key in self.__data_list]
        return pd.concat(data_list, ignore_index=True)
