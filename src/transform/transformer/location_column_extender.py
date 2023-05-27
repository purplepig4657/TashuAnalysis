from enum import Enum

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.base.rent_data_column_name import RentDataCN
from src.repository.data_loader import DataLoader


class StationDataColumnName:
    ID = 'id'
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'


class LocationColumnExtender(BaseEstimator, TransformerMixin):
    """
    Extend location data column.
    depends_on: :py:class:`src.transform.transformer.column_renamer.ColumnRenamer`
    """

    def __init__(self, year: str = "2021", only_rent_location: bool = False, data_loader: DataLoader = None):
        self.__data_loader = DataLoader() if data_loader is None else data_loader
        self.__only_rent_station = only_rent_location
        self.__station_data = self.__data_loader.get_specific_station_data(name=year)
        self.__location_data = self.__station_data[[
            StationDataColumnName.ID,
            StationDataColumnName.LATITUDE,
            StationDataColumnName.LONGITUDE
        ]]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return self.extend_location_data(X)

    def extend_location_data(self, data: pd.DataFrame) -> pd.DataFrame:
        rent_location_data_tmp = self.__location_data.rename(columns={
            StationDataColumnName.ID: RentDataCN.RENT_STATION,
            StationDataColumnName.LATITUDE: RentDataCN.RENT_LATITUDE,
            StationDataColumnName.LONGITUDE: RentDataCN.RENT_LONGITUDE
        })
        data = pd.merge(data, rent_location_data_tmp, on=RentDataCN.RENT_STATION)

        if not self.__only_rent_station:
            return_location_data_tmp = self.__location_data.rename(columns={
                StationDataColumnName.ID: RentDataCN.RETURN_STATION,
                StationDataColumnName.LATITUDE: RentDataCN.RETURN_LATITUDE,
                StationDataColumnName.LONGITUDE: RentDataCN.RETURN_LONGITUDE
            })
            data = pd.merge(data, return_location_data_tmp, on=RentDataCN.RETURN_STATION)

        data.sort_values(by=RentDataCN.RENT_DATE, ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data
