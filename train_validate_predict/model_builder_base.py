import abc


class ModelBuilderBase(abc.ABC):
    """
    Note: the class implementing this must not compile the tensorflow model.
    The compilation should be done by the trainer.
    """

    @abc.abstractmethod
    def build_model(self):
        """

        :return: True when build was success
        """
        pass

    @abc.abstractmethod
    def force_rebuild_model(self):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass
