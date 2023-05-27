import pandas as pd
from sklearn.model_selection import train_test_split


class RandomSampling:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame = None, test_size: float = 0.2, random_seed: int = 42):
        self.__X = X
        self.__y = y
        self.__test_size = test_size
        self.__random_seed = random_seed

    def train_test_split(self) -> (pd.DataFrame, pd.DataFrame):
        if self.__y is None:
            return train_test_split(self.__X, test_size=self.__test_size, random_state=self.__random_seed)
        else:
            return train_test_split(self.__X, self.__y, test_size=self.__test_size, random_state=self.__random_seed)
