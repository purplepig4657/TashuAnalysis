import os

import pandas as pd
import matplotlib.pyplot as plt


class RentalAmountScatter:
    """
    It is needless figure, but just example how to write eda.
    """

    def __init__(self, data: pd.DataFrame):
        self.__FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.__data = data

    def generate(self):
        self.__data.plot(kind='scatter', x='rent_longitude', y='rent_latitude',
                         s=self.__data['distance'] / 25, c='distance', figsize=(10, 7), alpha=0.1)
        plt.show()

    def to_file(self):
        self.__data.plot(kind='scatter', x='rent_longitude', y='rent_latitude',
                         s=self.__data['distance'] / 25, c='distance', figsize=(10, 7), alpha=0.1)
        self.save_figure(figure_name='rental_amount_scatter', path=f'{self.__FILE_PATH}/result')

    # noinspection PyMethodMayBeStatic
    def save_figure(self, figure_name, path, extension='png', resolution=300):
        file_path = os.path.join(path, f"{figure_name}.{extension}")
        plt.savefig(file_path, format=extension, dpi=resolution)
