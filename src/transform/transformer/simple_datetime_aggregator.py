import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import CN


class SimpleDatetimeAggregator(BaseEstimator, TransformerMixin):
    """
    Simple Aggregate data per datetime hour and per rent_station.
    depends_on: :py:class:`src.transform.transformer.column_renamer.ColumnRenamer`,
                :py:class:`src.transform.transformer.string_to_datetime_converter.StringToDatetimeConverter`
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        sampled_X = X[
            [CN.RENT_STATION, CN.RENT_DATE]
        ].copy()
        return self.aggregate(sampled_X)

    # noinspection PyMethodMayBeStatic
    def aggregate(self, X: pd.DataFrame) -> pd.DataFrame:
        X[CN.RENT_DATE] = X[CN.RENT_DATE].dt.floor('H')
        X[CN.RENT_COUNT] = X.groupby(CN.RENT_DATE)[CN.RENT_STATION].transform('count')
        X.drop_duplicates(subset=[CN.RENT_DATE, CN.RENT_STATION], inplace=True)
        return X
