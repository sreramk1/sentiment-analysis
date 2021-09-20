# Copyright (c) 2021 Sreram K (sreramk26@gmail.com), All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################


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
