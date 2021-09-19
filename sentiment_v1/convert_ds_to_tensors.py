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

from train_validate_predict.dataset_base import DataSetBase

from ds_specific_constants import TWEET_REVIEW_TXT_COLUMN
from ds_specific_constants import TWEET_LABEL_COLUMN


class TweetReviewDS(DataSetBase):

    def __init__(self,
                 pd_data_frame_ds: DataSetBase,
                 shuffle_buffer_size,
                 batch_size,
                 prefetch_size=tf.data.AUTOTUNE,
                 tweet_review_txt_column=TWEET_REVIEW_TXT_COLUMN,
                 tweet_label_column=TWEET_LABEL_COLUMN):

        self.__dataframe_ds = pd_data_frame_ds
        self.__shuffle_buffer_size = shuffle_buffer_size
        self.__batch_size = batch_size

        self.__prefetch_size = prefetch_size

        self.__twt_txt = tweet_review_txt_column
        self.__twt_lbl = tweet_label_column

        self.__train_ds = None
        self.__validate_ds = None

    def __convert_train_validate_ds_to_tensor(self, dataset_train, dataset_validate):
        ds_train_text_tf = tf.convert_to_tensor(dataset_train[self.__twt_txt], dtype=tf.string)
        ds_train_label_tf = tf.convert_to_tensor(dataset_train[self.__twt_lbl], dtype=tf.float32)

        ds_validate_text_tf = tf.convert_to_tensor(dataset_validate[self.__twt_txt], dtype=tf.string)
        ds_validate_label_tf = tf.convert_to_tensor(dataset_validate[self.__twt_lbl], dtype=tf.float32)

        ds_train_text_tf: tf.Tensor
        ds_train_label_tf: tf.Tensor
        ds_validate_text_tf: tf.Tensor
        ds_validate_label_tf: tf.Tensor

        return ds_train_text_tf, ds_train_label_tf, ds_validate_text_tf, ds_validate_label_tf

    @staticmethod
    def __convert_tensor_slices_to_dataset(ds_train_text_tf, ds_train_label_tf, ds_validate_text_tf,
                                           ds_validate_label_tf):
        train_ds = tf.data.Dataset.from_tensor_slices((ds_train_text_tf, ds_train_label_tf))
        validate_ds = tf.data.Dataset.from_tensor_slices((ds_validate_text_tf, ds_validate_label_tf))

        return train_ds, validate_ds

    def __shuffle_and_prepare_batches(self, train_ds, validate_ds):
        train_ds_res = train_ds.shuffle(self.__shuffle_buffer_size).batch(self.__batch_size).prefetch(
            self.__prefetch_size)
        validate_ds_res = validate_ds.batch(self.__shuffle_buffer_size).prefetch(self.__prefetch_size)

        return train_ds_res, validate_ds_res

    def __preprocess(self):
        if self.__train_ds is None or self.__validate_ds is None:
            dataset_train = self.__dataframe_ds.prepare_and_get_train_ds()
            dataset_validate = self.__dataframe_ds.prepare_and_get_validate_ds()

            ds_train_text_tf, ds_train_label_tf, ds_validate_text_tf, ds_validate_label_tf = \
                self.__convert_train_validate_ds_to_tensor(dataset_train, dataset_validate)

            train_ds, validate_ds = self.__convert_tensor_slices_to_dataset(ds_train_text_tf, ds_train_label_tf,
                                                                            ds_validate_text_tf, ds_validate_label_tf)

            self.__train_ds, self.__validate_ds = self.__shuffle_and_prepare_batches(train_ds, validate_ds)

    def prepare_and_get_train_ds(self):
        self.__preprocess()
        return self.__train_ds

    def prepare_and_get_validate_ds(self):
        self.__preprocess()
        return self.__validate_ds

    def reset(self):
        self.__train_ds = None
        self.__validate_ds = None


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()
    print(predictions)
