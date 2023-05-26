from src.eda.rental_amount_scatter.rental_amount_scatter import RentalAmountScatter
from src.repository.data_loader import DataLoader

data_loader = DataLoader()
data = data_loader.get_specific_rent_data("2021")

rental_amount_scatter = RentalAmountScatter(data=data)
rental_amount_scatter.generate_save(figure_name="rental_amount_scatter")
