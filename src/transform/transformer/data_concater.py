from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DataConcater(BaseEstimator, TransformerMixin):
    """
    Concatenating rent data.
    """

    def __init__(self, data_category: str = 'rent'):
        self.__data_category = data_category
        if data_category == 'rent':
            self.__concat_data_list = ["2016", "2017", "2018", "2019", "2020", "2021", "2022_1"]
        elif data_category == 'weather':
            self.__concat_data_list = ["2016", "2017", "2018", "2019", "2020", "2021", "2022"]

    def fit(self, X: dict, y=None):
        return self

    def transform(self, X: dict, y=None):
        return self.concatenate(X)

    def concatenate(self, X: dict):
        data_list = [value for key, value in X.items() if key in self.__concat_data_list]
        return pd.concat(data_list, ignore_index=True)
