from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import RentDataCN, WeatherDataCN


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom one hot encoder.
    """

    def __init__(self, encoding_data: list):
        self.__encoding_data = encoding_data

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return self.one_hot_encode(X)

    def one_hot_encode(self, X: pd.DataFrame) -> pd.DataFrame:
        one_hot_encoded_columns = list()

        for column in self.__encoding_data:
            one_hot_encoded_columns.append(pd.get_dummies(X[column], prefix=column))

        X = X.drop(self.__encoding_data, axis=1)
        X = pd.concat([X, *one_hot_encoded_columns], axis=1)

        X.columns = X.columns.astype(str)

        return X
