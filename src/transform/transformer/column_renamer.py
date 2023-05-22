from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class ColumnRenamer(BaseEstimator, TransformerMixin):
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
        return data.rename(columns={'대여스테이션': 'rent_station', '대여일시': 'rent_date', '반납스테이션': 'return_station',
                                    '반납일시': 'return_date', '이동거리': 'distance', '회원구분': 'user_category'})
