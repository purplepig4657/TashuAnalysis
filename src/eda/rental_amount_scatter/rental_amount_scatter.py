import pandas as pd
from sklearn.pipeline import Pipeline

from src.base.rent_data_column_name import RentDataCN
from src.eda.figure_eda_abstract import FigureEdaAbstract
from src.transform.transformer.column_renamer import ColumnRenamer
from src.transform.transformer.location_column_extender import LocationColumnExtender
from src.transform.transformer.string_to_datetime_converter import StringToDatetimeConverter


class RentalAmountScatter(FigureEdaAbstract):
    """
    It is needless figure, but just example show how to write eda.
    """

    def __init__(self, data: pd.DataFrame, is_processed: bool = False):
        self.__pipeline = Pipeline([
            ('renamer', ColumnRenamer()),
            ('str2datatime', StringToDatetimeConverter()),
            ('location_extender', LocationColumnExtender(only_rent_location=True))
        ])

        super().__init__(data=data, pipeline=self.__pipeline, is_processed=is_processed)

    def generate(self, data: pd.DataFrame):
        data.plot(
            kind='scatter',
            x=RentDataCN.RENT_LONGITUDE,
            y=RentDataCN.RENT_LATITUDE,
            s=data[RentDataCN.DISTANCE] / 25,
            c=RentDataCN.DISTANCE,
            figsize=(10, 7),
            alpha=0.1
        )
