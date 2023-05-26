import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
        sampled_X = X[['rent_station', 'rent_date']].copy()
        return self.aggregate(sampled_X)

    # noinspection PyMethodMayBeStatic
    def aggregate(self, X: pd.DataFrame) -> pd.DataFrame:
        X['rent_date'] = X['rent_date'].dt.floor('H')
        X['count'] = X.groupby('rent_date')['rent_station'].transform('count')
        X.drop_duplicates(subset=['rent_date', 'rent_station'], inplace=True)
        return X
