import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.repository.data_loader import DataLoader


class LocationColumnExtender(BaseEstimator, TransformerMixin):
    """
    Extend location data column.
    depends_on: :py:class:`src.transform.transformer.column_renamer.ColumnRenamer`
    """

    def __init__(self, year: str, only_rent_location: bool = False, data_loader: DataLoader = None):
        self.__data_loader = DataLoader() if data_loader is None else data_loader
        self.__only_rent_station = only_rent_location
        self.__station_data = self.__data_loader.get_specific_station_data(name=year)
        self.__location_data = self.__station_data[['id', 'latitude', 'longitude']]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return self.extend_location_data(X)

    def extend_location_data(self, data: pd.DataFrame) -> pd.DataFrame:
        rent_location_data_tmp = self.__location_data.rename(columns={
            'id': 'rent_station',
            'latitude': 'rent_latitude',
            'longitude': 'rent_longitude'
        })
        data = pd.merge(data, rent_location_data_tmp, on='rent_station')

        if not self.__only_rent_station:
            return_location_data_tmp = self.__location_data.rename(columns={
                'id': 'return_station',
                'latitude': 'return_latitude',
                'longitude': 'return_longitude'
            })
            data = pd.merge(data, return_location_data_tmp, on='return_station')

        data.sort_values(by='rent_date', ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data
