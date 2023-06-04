import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, WeatherDataCN, WeatherDataValue, TimeDataCN, TimeDataValue


class NighttimeDropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        return self

    # noinspection PyMethodMayBeStatic
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None):
        removed_X = X[X[TimeDataCN.TIME_CATEGORY] != TimeDataValue.NIGHTTIME]
        return removed_X
