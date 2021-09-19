import abc


class WeightsManagerBase:

    @abc.abstractmethod
    def set_model(self, model):
        pass

    @abc.abstractmethod
    def load_weights_from_file(self, path):
        pass

    @abc.abstractmethod
    def write_weights_to_file(self, path):
        pass

    @abc.abstractmethod
    def set_weights(self, weights):
        pass

    @abc.abstractmethod
    def get_weights(self):
        pass

    @abc.abstractmethod
    def set_weights_to_model(self):
        pass

    @abc.abstractmethod
    def read_weights_from_model(self):
        pass

