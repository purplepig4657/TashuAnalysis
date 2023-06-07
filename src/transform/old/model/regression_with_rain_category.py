from sklearn.pipeline import Pipeline

from src.base.column_name import RentDataCN, TimeDataCN, WeatherDataCN
from src.base.regression_model_base import RegressionModelBase
from src.repository.rent_data_loader import RentDataLoader
from src.repository.weather_data_loader import WeatherDataLoader
from src.transform.old.column_renamer import ColumnRenamer
from src.transform.old.data_concater import DataConcater
from src.transform.old.datetime_to_category import DatetimeToCategory
from src.transform.old.location_column_extender import LocationColumnExtender
from src.transform.common.custom_one_hot_encoder import CustomOneHotEncoder
from src.transform.old.rain_aggregator import RainAggregator
from src.transform.old.weather_column_extender import WeatherColumnExtender
from src.transform.old.string_to_datetime_converter import StringToDatetimeConverter
from src.transform.weather.weather_preprocessor import WeatherPreprocessor


class RegressionWithRainCategory(RegressionModelBase):
    def __init__(self):
        data_loader = RentDataLoader()
        weather_data_loader = WeatherDataLoader()

        weather_pipline = Pipeline([
            ('data_concatenate', DataConcater(data_category='weather')),
            ('renamer', ColumnRenamer()),
            ('str2datetime', StringToDatetimeConverter(data_category='weather')),
            ('preprocessor', WeatherPreprocessor())
        ])

        weather_data = weather_pipline.fit_transform(weather_data_loader.all_data)

        pipline = Pipeline([
            ('data_concatenate', DataConcater()),
            ('renamer', ColumnRenamer()),
            ('str2datatime', StringToDatetimeConverter(data_category='rent', per_hour=True)),
            ('location_extender', LocationColumnExtender(year="2021", only_rent_location=True)),
            ('weather_extender', WeatherColumnExtender(preprocessed_data=weather_data)),
            ('datetime2category', DatetimeToCategory()),
            ('aggregate', RainAggregator()),
            ('one-hot_encode', CustomOneHotEncoder([TimeDataCN.MONTH, TimeDataCN.DAY, TimeDataCN.WEEKDAY,
                                                    TimeDataCN.TIME_CATEGORY, WeatherDataCN.RAINFALL]))
        ])

        self.__processed_data = pipline.fit_transform(data_loader.all_data)

        self.__processed_data = self.__processed_data[self.__processed_data[RentDataCN.RENT_STATION] == 133]

        self.y = self.__processed_data[RentDataCN.RENT_COUNT]
        self.X = self.__processed_data.drop(columns=[RentDataCN.RENT_COUNT])

        super().__init__(self.X, self.y)

