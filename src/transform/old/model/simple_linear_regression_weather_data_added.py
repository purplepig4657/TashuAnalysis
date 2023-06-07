import pandas as pd
from sklearn.pipeline import Pipeline

from src.base.column_name import RentDataCN
from src.base.regression_model_base import RegressionModelBase
from src.repository.rent_data_loader import RentDataLoader
from src.repository.weather_data_loader import WeatherDataLoader
from src.transform.sampling.random_sampling import RandomSampling
from src.transform.old.column_renamer import ColumnRenamer
from src.transform.old.data_concater import DataConcater
from src.transform.old.location_column_extender import LocationColumnExtender
from src.transform.old.weather_column_extender import WeatherColumnExtender
from src.transform.old.simple_datetime_aggregator import SimpleDatetimeAggregator
from src.transform.old.string_to_datetime_converter import StringToDatetimeConverter
from src.transform.weather.weather_preprocessor import WeatherPreprocessor


class SimpleLinearRegressionWeatherDataAdded(RegressionModelBase):
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
            ('str2datatime', StringToDatetimeConverter()),
            ('location_extender', LocationColumnExtender(year="2021", only_rent_location=True)),
            ('aggregate', SimpleDatetimeAggregator()),
            ('weather_extender', WeatherColumnExtender(preprocessed_data=weather_data))
        ])

        self.__processed_data = pipline.fit_transform(data_loader.all_data)

        self.y = self.__processed_data[RentDataCN.RENT_COUNT]
        self.X = self.__processed_data.drop(columns=[RentDataCN.RENT_COUNT])
        self.X[RentDataCN.RENT_DATE] = self.X[RentDataCN.RENT_DATE].apply(lambda x: pd.to_datetime(x).timestamp())

        random_sampling = RandomSampling(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = random_sampling.train_test_split()

        super().__init__(self.X, self.y)
