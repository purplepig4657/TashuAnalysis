from sklearn.pipeline import Pipeline

from src.base.column_name import RentDataCN, TimeDataCN, WeatherDataCN, ClusterDataCN
from src.base.regression_model_base import RegressionModelBase
from src.repository.cluster_data_loader import ClusterDataLoader
from src.repository.rent_data_loader import RentDataLoader
from src.repository.weather_data_loader import WeatherDataLoader
from src.transform.cluster.cluster_column_renamer import ClusterColumnRenamer
from src.transform.cluster.cluster_data_aggregator import ClusterDataAggregator
from src.transform.cluster.cluster_extender import ClusterExtender
from src.transform.cluster.cluster_index_column_dropper import ClusterIndexColumnDropper
from src.transform.common.custom_one_hot_encoder import CustomOneHotEncoder
from src.transform.common.nighttime_dropper import NighttimeDropper
from src.transform.common.selected_column_dropper import SelectedColumnDropper
from src.transform.common.year_column_dropper import YearColumnDropper
from src.transform.rent.rent_column_renamer import RentColumnRenamer
from src.transform.rent.rent_concater import RentConcater
from src.transform.rent.rent_date_aggregator import RentDateAggregator
from src.transform.rent.rent_datetime_to_category_converter import RentDatetimeToCategoryConverter
from src.transform.rent.rent_hour_to_time_slot import RentHourToTimeSlot
from src.transform.rent.rent_preprocessor import RentPreprocessor
from src.transform.rent.rent_string_to_datetime_converter import RentStringToDatetimeConverter
from src.transform.weather.weather_column_renamer import WeatherColumnRenamer
from src.transform.weather.weather_concater import WeatherConcater
from src.transform.weather.weather_date_aggregator import WeatherDateAggregator
from src.transform.weather.weather_datetime_to_category_converter import WeatherDatetimeToCategoryConverter
from src.transform.weather.weather_extender import WeatherExtender
from src.transform.weather.weather_hour_to_time_slot import WeatherHourToTimeSlot
from src.transform.weather.weather_string_to_datetime_converter import WeatherStringToDatetimeConverter
from src.transform.weather.weather_preprocessor import WeatherPreprocessor


class RegressionWeatherTimeslotCluster(RegressionModelBase):
    def __init__(self, cluster_data: str = '5'):
        rent_data_loader = RentDataLoader()
        weather_data_loader = WeatherDataLoader()
        cluster_data_loader = ClusterDataLoader()

        is_categorized = True

        cluster_pipline = Pipeline([
            ('rename', ClusterColumnRenamer()),
            ('index_column_drop', ClusterIndexColumnDropper())
        ])

        processed_cluster_data = cluster_pipline.fit_transform(cluster_data_loader.get_specific_data(cluster_data))

        weather_pipline = Pipeline([
            ('data_concatenate', WeatherConcater()),
            ('rename', WeatherColumnRenamer()),
            ('str2datetime', WeatherStringToDatetimeConverter()),
            ('preprocessing', WeatherPreprocessor()),
            ('datetime2category', WeatherDatetimeToCategoryConverter()),
            ('hour2timeslot', WeatherHourToTimeSlot()),
            ('aggregator', WeatherDateAggregator(is_categorized=is_categorized))
        ])

        processed_weather_data = weather_pipline.fit_transform(weather_data_loader.all_data)

        rent_pipline = Pipeline([
            ('data_concatenate', RentConcater()),
            ('rename', RentColumnRenamer()),
            ('str2datetime', RentStringToDatetimeConverter()),
            ('preprocessing', RentPreprocessor()),
            ('datetime2category', RentDatetimeToCategoryConverter()),
            ('hour2timeslot', RentHourToTimeSlot()),
            ('aggregator', RentDateAggregator(is_categorized=is_categorized)),
            ('weather_extender', WeatherExtender(preprocessed_data=processed_weather_data, is_categorized=is_categorized)),
            ('cluster_extender', ClusterExtender(cluster_data=processed_cluster_data)),
            ('cluster_aggregator', ClusterDataAggregator(is_categorized=is_categorized)),
            ('year_drop', YearColumnDropper()),
            ('nighttime_drop', NighttimeDropper()),
            ('selected_column_drop', SelectedColumnDropper(selected_columns=[WeatherDataCN.RAINFALL]))
        ])

        self.__processed_data = rent_pipline.fit_transform(rent_data_loader.all_data)

        # self.__processed_data = self.__processed_data[self.__processed_data[ClusterDataCN.CLUSTER] == 3]

        custom_one_hot_encoder = CustomOneHotEncoder([TimeDataCN.MONTH, TimeDataCN.DAY, TimeDataCN.WEEKDAY,
                                                      TimeDataCN.TIME_CATEGORY, ClusterDataCN.CLUSTER])

        self.__processed_data = custom_one_hot_encoder.fit_transform(self.__processed_data)

        self.y = self.__processed_data[RentDataCN.RENT_COUNT]
        self.X = self.__processed_data.drop(columns=[RentDataCN.RENT_COUNT])

        super().__init__(self.X, self.y)
