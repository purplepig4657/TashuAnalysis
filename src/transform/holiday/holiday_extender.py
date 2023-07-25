import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, WeatherDataCN, TimeDataCN


class HolidayExtender(BaseEstimator, TransformerMixin):
    """
    Extend weather data column.
    """

    def __init__(self, preprocessed_data: pd.DataFrame, is_categorized: bool = False):
        if preprocessed_data is None:
            raise "Required data is none."

        self.__holiday_data = preprocessed_data

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return self.extend_weather_data(X)

    def extend_weather_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = pd.merge(data, self.__holiday_data, on=[
            TimeDataCN.YEAR,
            TimeDataCN.MONTH,
            TimeDataCN.DAY
        ])

        # data.sort_values(by=RentDataCN.RENT_DATE, ascending=True, inplace=True)
        # data.reset_index(drop=True, inplace=True)

        return data
