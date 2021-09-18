import abc


class DsReader(abc.ABC):

    @abc.abstractmethod
    def load_dataset(self):
        pass

    @abc.abstractmethod
    def get_dataset(self):
        pass
