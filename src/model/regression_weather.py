from sklearn.pipeline import Pipeline

from src.base.column_name import RentDataCN, TimeDataCN, WeatherDataCN
from src.base.regression_model_base import RegressionModelBase
from src.repository.rent_data_loader import RentDataLoader
from src.repository.weather_data_loader import WeatherDataLoader
from src.transform.common.custom_one_hot_encoder import CustomOneHotEncoder
from src.transform.common.year_column_dropper import YearColumnDropper
from src.transform.rent.rent_column_renamer import RentColumnRenamer
from src.transform.rent.rent_concater import RentConcater
from src.transform.rent.rent_date_aggregator import RentDateAggregator
from src.transform.rent.rent_datetime_to_category_converter import RentDatetimeToCategoryConverter
from src.transform.rent.rent_preprocessor import RentPreprocessor
from src.transform.rent.rent_string_to_datetime_converter import RentStringToDatetimeConverter
from src.transform.weather.weather_column_renamer import WeatherColumnRenamer
from src.transform.weather.weather_concater import WeatherConcater
from src.transform.weather.weather_date_aggregator import WeatherDateAggregator
from src.transform.weather.weather_datetime_to_category_converter import WeatherDatetimeToCategoryConverter
from src.transform.weather.weather_extender import WeatherExtender
from src.transform.weather.weather_preprocessor import WeatherPreprocessor
from src.transform.weather.weather_string_to_datetime_converter import WeatherStringToDatetimeConverter


class RegressionWeather(RegressionModelBase):
    def __init__(self):
        rent_data_loader = RentDataLoader()
        weather_data_loader = WeatherDataLoader()

        weather_pipline = Pipeline([
            ('data_concatenate', WeatherConcater()),
            ('rename', WeatherColumnRenamer()),
            ('str2datetime', WeatherStringToDatetimeConverter()),
            ('preprocessing', WeatherPreprocessor()),
            ('datetime2category', WeatherDatetimeToCategoryConverter()),
            ('aggregator', WeatherDateAggregator())
        ])

        processed_weather_data = weather_pipline.fit_transform(weather_data_loader.all_data)

        rent_pipline = Pipeline([
            ('data_concatenate', RentConcater()),
            ('rename', RentColumnRenamer()),
            ('str2datetime', RentStringToDatetimeConverter()),
            ('preprocessing', RentPreprocessor()),
            ('datetime2category', RentDatetimeToCategoryConverter()),
            ('aggregator', RentDateAggregator()),
            ('weather_extend', WeatherExtender(preprocessed_data=processed_weather_data)),
            ('year_drop', YearColumnDropper()),
        ])

        self.__processed_data = rent_pipline.fit_transform(rent_data_loader.all_data)

        custom_one_hot_encoder = CustomOneHotEncoder([TimeDataCN.MONTH, TimeDataCN.DAY, TimeDataCN.WEEKDAY,
                                                      TimeDataCN.HOUR, WeatherDataCN.RAINFALL])

        self.__processed_data = custom_one_hot_encoder.fit_transform(self.__processed_data)

        # self.__processed_data = self.__processed_data[self.__processed_data[RentDataCN.RENT_STATION] == 133]

        self.y = self.__processed_data[RentDataCN.RENT_COUNT]
        self.X = self.__processed_data.drop(columns=[RentDataCN.RENT_COUNT])

        super().__init__(self.X, self.y)
