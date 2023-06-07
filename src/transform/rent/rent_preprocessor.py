from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class RentPreprocessor(BaseEstimator, TransformerMixin):
    """
    Rent data missed value preprocessing.
    """

    def __init__(self):
        pass

    def fit(self, X: dict, y=None):
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X
