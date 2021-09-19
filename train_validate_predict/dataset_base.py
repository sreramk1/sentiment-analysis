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
