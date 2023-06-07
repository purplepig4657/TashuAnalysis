from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import LocationDataCN


class ClusterColumnRenamer(BaseEstimator, TransformerMixin):
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
            'id': LocationDataCN.ID,
            'version': LocationDataCN.VERSION,
            'station_name': LocationDataCN.STATION_NAME,
            'gu': LocationDataCN.GU,
            'dong': LocationDataCN.DONG,
            'latitude': LocationDataCN.LATITUDE,
            'longitude': LocationDataCN.LONGITUDE
        })
