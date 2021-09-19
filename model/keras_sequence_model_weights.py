import json

from model.weights_manager_abst import WeightsManagerBase
from model.util.numpy_encoder_json import NdarrayEncoder
from model.util.convert_to_ndarray import convert_to_ndarray


class KerasSequenceModelWeights(WeightsManagerBase):

    def __init__(self):
        self.__model = None

        self.__weights_by_layer = None

    def set_model(self, model):
        self.__model = model

    def load_weights_from_file(self, path):
        with open(path, "r") as json_file_r:
            weights_as_json = json_file_r.read()
        weights_by_layer = json.loads(weights_as_json)
        self.__weights_by_layer = convert_to_ndarray(weights_by_layer)
        return self.__weights_by_layer

    def write_weights_to_file(self, path):
        weights_in_json = json.dumps(self.__weights_by_layer, cls=NdarrayEncoder)
        with open(path, "w") as json_file:
            json_file.write(weights_in_json)

    def set_weights(self, weights):
        self.__weights_by_layer = weights

    def get_weights(self):
        return self.__weights_by_layer

    def set_weights_to_model(self):
        i = 0
        for layer in self.__model.layers:
            layer.set_weights(self.__weights_by_layer[i])
            i += 1

    def read_weights_from_model(self):
        self.__weights_by_layer = []
        for layer in self.__model.layers:
            self.__weights_by_layer.append(layer.get_weights())
