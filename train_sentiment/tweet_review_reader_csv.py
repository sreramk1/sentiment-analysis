import pandas as pd

from reader import DsReader


class TweetReviewReaderCSV(DsReader):

    def __init__(self, csv_path):
        self.__csv_path = csv_path
        self.__dataset = None

    def load_dataset(self):
        self.__dataset = pd.read_csv(self.__csv_path)

    def get_dataset(self):
        if self.__dataset is None:
            raise Exception("Forbidden call. Dataset was uninitialized")

