from sklearn.pipeline import Pipeline

from src.base.column_name import RentDataCN, TimeDataCN
from src.base.regression_model_base import RegressionModelBase
from src.repository.rent_data_loader import RentDataLoader
from src.repository.weather_data_loader import WeatherDataLoader
from src.transform.common.column_renamer import ColumnRenamer
from src.transform.common.data_concater import DataConcater
from src.transform.common.datetime_to_category import DatetimeToCategory
from src.transform.location.location_column_extender import LocationColumnExtender
from src.transform.common.custom_one_hot_encoder import CustomOneHotEncoder
from src.transform.common.nighttime_dropper import NighttimeDropper
from src.transform.common.specific_time_slot_aggregator import SpecificTimeSlotAggregator
from src.transform.weather.weather_column_extender import WeatherColumnExtender
from src.transform.common.string_to_datetime_converter import StringToDatetimeConverter
from src.transform.weather.weather_data_preprocessor import WeatherDataPreprocessor


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
            ('dropper', NighttimeDropper()),
            ('one-hot_encode', CustomOneHotEncoder([TimeDataCN.MONTH, TimeDataCN.DAY,
                                                    TimeDataCN.WEEKDAY, TimeDataCN.TIME_CATEGORY]))
        ])

        self.__processed_data = pipline.fit_transform(data_loader.all_data)

        self.__processed_data = self.__processed_data[self.__processed_data[RentDataCN.RENT_STATION] == 133]

        self.y = self.__processed_data[RentDataCN.RENT_COUNT]
        self.X = self.__processed_data.drop(columns=[RentDataCN.RENT_COUNT])

        super().__init__(self.X, self.y)
