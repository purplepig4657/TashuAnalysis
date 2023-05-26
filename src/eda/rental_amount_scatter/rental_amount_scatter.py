import os

import pandas as pd

from src.eda.figure_eda_abstract import FigureEdaAbstract


class RentalAmountScatter(FigureEdaAbstract):
    """
    It is needless figure, but just example how to write eda.
    """

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.__FILE_PATH = os.path.dirname(os.path.abspath(__file__))

    def generate(self, data: pd.DataFrame):
        data.plot(kind='scatter', x='rent_longitude', y='rent_latitude',
                  s=data['distance'] / 25, c='distance', figsize=(10, 7), alpha=0.1)
