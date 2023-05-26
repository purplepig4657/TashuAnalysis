from sklearn.pipeline import Pipeline

from src.eda.rental_amount_scatter.rental_amount_scatter import RentalAmountScatter
from src.repository.data_loader import DataLoader
from src.transform.transformer.column_renamer import ColumnRenamer
from src.transform.transformer.location_column_extender import LocationColumnExtender
from src.transform.transformer.string_to_datetime_converter import StringToDatetimeConverter

data_loader = DataLoader()

pipline = Pipeline([
    ('renamer', ColumnRenamer()),
    ('str2datatime', StringToDatetimeConverter()),
    ('location_extender', LocationColumnExtender(year="2021", only_rent_location=True, data_loader=data_loader))
])

data = data_loader.get_specific_rent_data("2021")
processed_data = pipline.fit_transform(data)

rental_amount_scatter = RentalAmountScatter(processed_data)

rental_amount_scatter.generate()
rental_amount_scatter.to_file()
