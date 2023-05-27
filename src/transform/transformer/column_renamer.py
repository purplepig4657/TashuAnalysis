from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import CN


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
        return data.rename(columns={
            '대여스테이션': CN.RENT_STATION,
            '대여일시': CN.RENT_DATE,
            '반납스테이션': CN.RETURN_STATION,
            '반납일시': CN.RETURN_DATE,
            '이동거리': CN.DISTANCE,
            '회원구분': CN.USER_CATEGORY
        })
