import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.repository.data_loader import DataLoader
from src.base.column_name import CN
from src.transform.transformer.rent_data_concater import RentDataConcater
from src.transform.transformer.simple_datetime_aggregator import SimpleDatetimeAggregator
from src.transform.sampling.random_sampling import RandomSampling
from src.transform.transformer.location_column_extender import LocationColumnExtender
from src.transform.transformer.column_renamer import ColumnRenamer
from src.transform.transformer.string_to_datetime_converter import StringToDatetimeConverter


class SimpleLinearRegression:
    def __init__(self):
        self.__lin_reg = LinearRegression()

        data_loader = DataLoader()

        pipline = Pipeline([
            ('data_concatenate', RentDataConcater()),
            ('renamer', ColumnRenamer()),
            ('str2datatime', StringToDatetimeConverter()),
            ('location_extender', LocationColumnExtender(year="2021", only_rent_location=True, data_loader=data_loader)),
            ('aggregate', SimpleDatetimeAggregator())
        ])

        self.__processed_data = pipline.fit_transform(data_loader.all_rent_data)

        self.y = self.__processed_data[CN.RENT_COUNT]
        self.X = self.__processed_data.drop(columns=[CN.RENT_COUNT])
        self.X[CN.RENT_DATE] = self.X[CN.RENT_DATE].apply(lambda x: pd.to_datetime(x).timestamp())

        random_sampling = RandomSampling(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = random_sampling.train_test_split()

    def fit(self):
        self.__lin_reg.fit(self.X_train, self.y_train)

    def rmse(self):
        test_predictions = self.__lin_reg.predict(self.X_test)
        lin_mse = mean_squared_error(self.y_test, test_predictions)
        lin_rmse = np.sqrt(lin_mse)
        return lin_rmse
