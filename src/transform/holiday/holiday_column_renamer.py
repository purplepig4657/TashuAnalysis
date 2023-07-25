from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import HolidayDataCN


class HolidayColumnRenamer(BaseEstimator, TransformerMixin):
    """
    Renaming column to english.
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return self.renaming(X)

    # noinspection PyMethodMayBeStatic
    def renaming(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.rename(columns={
            'holiday_date': HolidayDataCN.DATE
        })
