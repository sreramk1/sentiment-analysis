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

import tensorflow as tf

from train_validate_predict.model_builder_base import ModelBuilderBase
from train_validate_predict.dataset_base import DataSetBase


class SentimentLSTMDense64Dense32Dense16Dense8Dense1(ModelBuilderBase):

    def __init__(self,
                 tf_ds_store: DataSetBase,
                 max_tokens=None):
        self.__max_tokens = max_tokens
        self.__tf_ds_store = tf_ds_store
        self.__encoder = None
        self.__model = None

    def __clear(self):
        self.__encoder = None

    def __create_encoder(self):
        self.__encoder = tf.keras.layers.TextVectorization(max_tokens=self.__max_tokens)
        self.__encoder.adapt(self.__tf_ds_store.prepare_and_get_train_ds().map(lambda text, label: text))
        return self.__encoder

    def build_model(self):
        if self.__encoder is None:
            encoder = self.__create_encoder()

            embedding_vocabulary_size = len(encoder.get_vocabulary())
            embedding_output_dimensions = 64
            mask_zero = True

            dense_layer_1_units = 64
            dense_layer_2_units = 32
            dense_layer_3_units = 16
            dense_layer_4_units = 8
            dense_layer_5_output_layer_units = 1

            dense_activation = "relu"

            self.__model = tf.keras.Sequential([
                encoder,
                tf.keras.layers.Embedding(
                    input_dim=embedding_vocabulary_size,
                    output_dim=embedding_output_dimensions,
                    # Use masking to handle the variable sequence lengths
                    mask_zero=mask_zero),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

                tf.keras.layers.Dense(dense_layer_1_units, activation=dense_activation),
                tf.keras.layers.Dense(dense_layer_2_units, activation=dense_activation),
                tf.keras.layers.Dense(dense_layer_3_units, activation=dense_activation),
                tf.keras.layers.Dense(dense_layer_4_units, activation=dense_activation),
                tf.keras.layers.Dense(dense_layer_5_output_layer_units)  # , activation='sigmoid')
            ])
            return True

        return False

    def force_rebuild_model(self):
        self.__encoder = None
        self.build_model()

    def get_model(self):
        return self.__model
