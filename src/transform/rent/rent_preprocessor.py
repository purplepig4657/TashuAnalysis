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
        X = self.drop_nan_return_station(X)
        X = self.drop_zero_station(X)
        X = self.drop_distance_column(X)
        X = self.drop_recent_data(X)
        X = self.drop_abnormal_data(X)
        return X

    # noinspection PyMethodMayBeStatic
    def drop_nan_rent_station(self, X: pd.DataFrame) -> pd.DataFrame:
        X.dropna(subset=[RentDataCN.RENT_STATION], inplace=True)
        return X
    
    # noinspection PyMethodMayBeStatic
    def drop_nan_return_station(self, X: pd.DataFrame) -> pd.DataFrame:
        X.dropna(subset=[RentDataCN.RETURN_STATION], inplace=True)
        return X
    
    # noinspection PyMethodMayBeStatic
    def drop_zero_station(self, X: pd.DataFrame) -> pd.DataFrame:
        X.drop(X[(X[RentDataCN.RENT_STATION] == 0) | (X[RentDataCN.RETURN_STATION] == 0)].index, inplace=True)
        return X
    
    # noinspection PyMethodMayBeStatic
    def drop_distance_column(self, X: pd.DataFrame) -> pd.DataFrame:
        X.drop(labels=RentDataCN.DISTANCE, axis=1, inplace=True)
        return X
    
    # noinspection PyMethodMayBeStatic
    def drop_recent_data(self, X: pd.DataFrame) -> pd.DataFrame:
        X.drop(X[(X[RentDataCN.RENT_STATION] > 262) | (X[RentDataCN.RETURN_STATION] > 262)].index, inplace=True)
        return X
    
    # noinspection PyMethodMayBeStatic
    def drop_abnormal_data(self, X: pd.DataFrame) -> pd.DataFrame:
        X.drop(X[(X[RentDataCN.RENT_STATION] == X[RentDataCN.RETURN_STATION]) &
                 (X[RentDataCN.RETURN_DATE] - X[RentDataCN.RENT_DATE] <= pd.Timedelta(minutes=5))].index, inplace=True)
        return X
