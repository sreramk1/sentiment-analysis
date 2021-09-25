from sentiment_v1.model_builder import SentimentLSTMDense64Dense32Dense16Dense8Dense1
from train_validate_predict.model_predict import ModelPredict
from model.keras_sequence_model_weights import KerasSequenceModelWeights


class ModelBuildAndPredict:
    """

    """

    def __init__(self, weights_file_path):
        self.__weights_file_path = weights_file_path
        self.__model_predict = None

    def build_model(self):
        sequence_model_weights = KerasSequenceModelWeights()
        loaded_weights = sequence_model_weights.load_weights_from_file(
            self.__weights_file_path)

        model_builder = SentimentLSTMDense64Dense32Dense16Dense8Dense1(layered_weights=loaded_weights)
        model_builder.build_model()
        model = model_builder.get_model()

        self.__model_predict = ModelPredict(model)

    def predict(self, input_data):
        return self.__model_predict.predict(input_data)