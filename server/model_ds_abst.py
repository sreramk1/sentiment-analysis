import abc


class DataSetBase(abc.ABC):

    @abc.abstractmethod
    def prepare_and_get_train_ds(self):
        pass

    @abc.abstractmethod
    def prepare_and_get_validate_ds(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """
        reset resets the dataset, clearing all the internal caches.
        :return:
        """
        pass


class ModelBuilderBase(abc.ABC):
    @abc.abstractmethod
    def build_model(self):
        pass


class TrainValidateBase(abc.ABC):

    @abc.abstractmethod
    def set_config(self, config):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, input_data) -> float:
        pass


if __name__ == "__main__":
    from abc import ABC


    class MyABC(ABC):
        @abc.abstractmethod
        def __init__(self, arg):
            self.__arg = arg
            self.__arg1 = None
            self.__arg2 = None

        @abc.abstractmethod
        def meth2(self, arg1, arg2):
            pass

        pass


    MyABC.register(tuple)

    assert issubclass(tuple, MyABC)
    assert isinstance((), MyABC)


    class NewClass(MyABC):
        def __init__(self, arg1, arg2):
            self.__arg1 = arg1
            self.__arg2 = arg2
            pass

        def initialize(self):
            pass

        def meth2(self, arg1):
            self.__arg1 = arg1
            # self.__arg2 = arg2


    x = NewClass(1, 2)
