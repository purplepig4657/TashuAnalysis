import datetime

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.base.column_name import HolidayDataCN


class HolidayDatetimeFilling(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        holiday_date = X[[HolidayDataCN.DATE]].values
        X = self.datetime_filling()
        X[HolidayDataCN.DATE] = pd.to_datetime(X[HolidayDataCN.DATE])
        X = self.extend_is_holiday_column(X, holiday_date)
        return X

    # noinspection PyMethodMayBeStatic
    def datetime_filling(self) -> pd.DataFrame:
        start_date = datetime.datetime(2016, 1, 1)
        end_date = datetime.datetime(2022, 12, 31)

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        return pd.DataFrame(date_range, columns=[HolidayDataCN.DATE])

    # noinspection PyMethodMayBeStatic
    def extend_is_holiday_column(self, X: pd.DataFrame, holiday_date) -> pd.DataFrame:
        X[HolidayDataCN.IS_HOLIDAY] = False
        for i in range(len(X)):
            weekday = X.loc[i, HolidayDataCN.DATE].weekday()
            if X.loc[i, HolidayDataCN.DATE] in holiday_date or weekday == 5 or weekday == 6:
                X.loc[i, HolidayDataCN.IS_HOLIDAY] = True

        return X
