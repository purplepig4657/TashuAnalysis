import pandas as pd
from sklearn.pipeline import Pipeline

from src.base.column_name import RentDataCN, TimeDataCN
from src.base.regression_model_base import RegressionModelBase
from src.repository.rent_data_loader import RentDataLoader
from src.repository.weather_data_loader import WeatherDataLoader
from src.transform.transformer.column_renamer import ColumnRenamer
from src.transform.transformer.data_concater import DataConcater
from src.transform.transformer.datetime_to_category import DatetimeToCategory
from src.transform.transformer.location_column_extender import LocationColumnExtender
from src.transform.transformer.specific_time_slot_aggregator import SpecificTimeSlotAggregator
from src.transform.transformer.weather_column_extender import WeatherColumnExtender
from src.transform.transformer.string_to_datetime_converter import StringToDatetimeConverter
from src.transform.transformer.weather_data_preprocessor import WeatherDataPreprocessor


class RegressionWithTimeCategory(RegressionModelBase):
    def __init__(self):
        data_loader = RentDataLoader()
        weather_data_loader = WeatherDataLoader()

        weather_pipline = Pipeline([
            ('data_concatenate', DataConcater(data_category='weather')),
            ('renamer', ColumnRenamer()),
            ('str2datetime', StringToDatetimeConverter(data_category='weather')),
            ('preprocessor', WeatherDataPreprocessor())
        ])

        weather_data = weather_pipline.fit_transform(weather_data_loader.all_data)

        pipline = Pipeline([
            ('data_concatenate', DataConcater()),
            ('renamer', ColumnRenamer()),
            ('str2datatime', StringToDatetimeConverter(data_category='rent', per_hour=True)),
            ('location_extender', LocationColumnExtender(year="2021", only_rent_location=True)),
            ('weather_extender', WeatherColumnExtender(preprocessed_data=weather_data)),
            ('datetime2category', DatetimeToCategory()),
            ('aggregate', SpecificTimeSlotAggregator()),
        ])

        self.__processed_data = pipline.fit_transform(data_loader.all_data)

        self.__category_columns = [TimeDataCN.MONTH, TimeDataCN.DAY, TimeDataCN.WEEKDAY, TimeDataCN.TIME_CATEGORY]

        self.__one_hot_encoded_columns = list()

        for column in self.__category_columns:
            self.__one_hot_encoded_columns.append(pd.get_dummies(self.__processed_data[column]))

        self.__processed_data = self.__processed_data.drop(self.__category_columns, axis=1)
        self.__processed_data = pd.concat([self.__processed_data, *self.__one_hot_encoded_columns], axis=1)

        self.__processed_data.columns = self.__processed_data.columns.astype(str)

        self.__processed_data = self.__processed_data[self.__processed_data[RentDataCN.RENT_STATION] == 133]

        self.y = self.__processed_data[RentDataCN.RENT_COUNT]
        self.X = self.__processed_data.drop(columns=[RentDataCN.RENT_COUNT])

        super().__init__(self.X, self.y)
