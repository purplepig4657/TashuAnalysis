import os
from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


class FigureEdaAbstract(ABC):
    def __init__(self, data: pd.DataFrame, pipeline: Pipeline, is_processed: bool = False):
        self.__CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
        self.__PATH = os.path.join(self.__CURRENT_PATH, 'result')

        self.__data = data if is_processed else pipeline.fit_transform(data)

    def to_figure(self):
        self.generate(self.__data)
        plt.show()

    def to_file(self, figure_name: str, extension='png', resolution=300):
        self.generate(self.__data)
        self.save_figure(figure_name=figure_name, extension=extension, resolution=resolution)

    def generate_save(self, figure_name: str, extension='png', resolution=300):
        self.to_figure()
        self.to_file(figure_name=figure_name, extension=extension, resolution=resolution)

    def save_figure(self, figure_name: str, extension: str, resolution: int):
        file_path = os.path.join(self.__PATH, f"{figure_name}.{extension}")
        plt.savefig(file_path, format=extension, dpi=resolution)

    @abstractmethod
    def generate(self, data: pd.DataFrame):
        pass
