import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.column_name import RentDataCN, WeatherDataCN


class WeatherColumnExtender(BaseEstimator, TransformerMixin):
    """
    Extend weather data column.
    depends_on: :py:class:`src.transform.transformer.column_renamer.ColumnRenamer`
    """

    def __init__(self, preprocessed_data: pd.DataFrame):
        if preprocessed_data is None:
            raise "Required data is none."
        self.__weather_data = preprocessed_data
        self.__weather_data = self.__weather_data[[
            WeatherDataCN.DATE,
            WeatherDataCN.TEMPERATURE,
            WeatherDataCN.PRECIPITATION,
            WeatherDataCN.SUNSHINE_DURATION
        ]]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return self.extend_weather_data(X)

    def extend_weather_data(self, data: pd.DataFrame) -> pd.DataFrame:
        self.__weather_data = self.__weather_data.rename(columns={
            WeatherDataCN.DATE: RentDataCN.RENT_DATE
        })

        data = pd.merge(data, self.__weather_data, on=RentDataCN.RENT_DATE)

        data.sort_values(by=RentDataCN.RENT_DATE, ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data

