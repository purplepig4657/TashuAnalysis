from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import RentDataCN


class RentPreprocessor(BaseEstimator, TransformerMixin):
    """
    Rent data missed value preprocessing.
    """

    def __init__(self):
        pass

    def fit(self, X: dict, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = self.drop_nan_rent_station(X)
        return X

    # noinspection PyMethodMayBeStatic
    def drop_nan_rent_station(self, X: pd.DataFrame) -> pd.DataFrame:
        X.dropna(subset=[RentDataCN.RENT_STATION], inplace=True)
        return X
