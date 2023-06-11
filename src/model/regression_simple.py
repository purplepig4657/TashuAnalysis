from sklearn.pipeline import Pipeline

from src.base.column_name import RentDataCN, TimeDataCN
from src.base.regression_model_base import RegressionModelBase
from src.repository.rent_data_loader import RentDataLoader
from src.transform.common.custom_one_hot_encoder import CustomOneHotEncoder
from src.transform.rent.rent_column_renamer import RentColumnRenamer
from src.transform.rent.rent_concater import RentConcater
from src.transform.rent.rent_date_aggregator import RentDateAggregator
from src.transform.rent.rent_datetime_to_category_converter import RentDatetimeToCategoryConverter
from src.transform.rent.rent_preprocessor import RentPreprocessor
from src.transform.rent.rent_string_to_datetime_converter import RentStringToDatetimeConverter


class RegressionSimple(RegressionModelBase):
    def __init__(self):
        rent_data_loader = RentDataLoader()

        rent_pipline = Pipeline([
            ('data_concatenate', RentConcater()),
            ('rename', RentColumnRenamer()),
            ('str2datetime', RentStringToDatetimeConverter()),
            ('preprocessing', RentPreprocessor()),
            ('datetime2category', RentDatetimeToCategoryConverter()),
            ('aggregator', RentDateAggregator())
        ])

        self.__processed_data = rent_pipline.fit_transform(rent_data_loader.all_data)

        custom_one_hot_encoder = CustomOneHotEncoder([TimeDataCN.YEAR, TimeDataCN.MONTH, TimeDataCN.DAY,
                                                      TimeDataCN.WEEKDAY, TimeDataCN.HOUR])

        self.__processed_data = custom_one_hot_encoder.fit_transform(self.__processed_data)

        # self.__processed_data = self.__processed_data[self.__processed_data[RentDataCN.RENT_STATION] == 133]

        self.y = self.__processed_data[RentDataCN.RENT_COUNT]
        self.X = self.__processed_data.drop(columns=[RentDataCN.RENT_COUNT])

        super().__init__(self.X, self.y)
