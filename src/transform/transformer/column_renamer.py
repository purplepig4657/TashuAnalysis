from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.rent_data_column_name import RentDataCN


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
            '대여스테이션': RentDataCN.RENT_STATION,
            '대여일시': RentDataCN.RENT_DATE,
            '반납스테이션': RentDataCN.RETURN_STATION,
            '반납일시': RentDataCN.RETURN_DATE,
            '이동거리': RentDataCN.DISTANCE,
            '회원구분': RentDataCN.USER_CATEGORY
        })
