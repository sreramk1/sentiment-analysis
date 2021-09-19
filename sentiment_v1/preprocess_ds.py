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

import pandas as pd
from sklearn.model_selection import train_test_split

from train_validate_predict.dataset_base import DataSetBase

from ds_specific_constants import POSITIVE_LABEL
from ds_specific_constants import NEGATIVE_LABEL
from ds_specific_constants import TWEET_REVIEW_TXT_COLUMN
from ds_specific_constants import TWEET_LABEL_COLUMN

from train_validate_predict.reader import DsReader


class TweetReviewToPdDataFrame(DataSetBase):

    def __init__(self,
                 ds_reader: DsReader,
                 positive_label=POSITIVE_LABEL,
                 negative_label=NEGATIVE_LABEL,
                 tweet_review_txt_column=TWEET_REVIEW_TXT_COLUMN,
                 tweet_label_column=TWEET_LABEL_COLUMN,
                 train_validate_split_ratio=0.2,
                 ):

        self.__ds_reader = ds_reader

        self.__p_lbl = positive_label
        self.__n_lbl = negative_label
        self.__twt_txt = tweet_review_txt_column
        self.__twt_lbl = tweet_label_column

        self.__train_validate_split_ratio = train_validate_split_ratio

        self.__dataset = None
        self.__dataset_processed = None

        self.__dataset_processed: pd.DataFrame

        self.__train_ds = None
        self.__validate_ds = None

    @staticmethod
    def __gen_strip_words_starting_with_filter(filter_str):
        def operation(txt2):
            return ' '.join(word for word in txt2.split(' ') if not word.startswith(filter_str))

        return operation

    @staticmethod
    def __gen_filter_numbers(replace_with):

        def operation(txt2):
            no_digits = []
            for ch in txt2:
                if not ch.isdigit():
                    no_digits.append(ch)
                else:
                    no_digits.append(replace_with)
            return ''.join(no_digits)

        return operation

    def __gen_process_label(self):
        def process(label):
            if label == self.__n_lbl:
                return 0
            elif label == self.__p_lbl:
                return 1
            raise Exception("unrecognized label")

        return process

    def __preprocess_dataset(self):

        if self.__dataset is None:
            raise Exception("expected __dataset to be set, before a call to __preprocess_dataset()")

        self.__dataset_processed = pd.DataFrame.copy(self.__dataset, deep=True)

        self.__dataset_processed[self.__twt_txt] = self.__dataset_processed[self.__twt_txt].apply(
            TweetReviewToPdDataFrame.__gen_strip_words_starting_with_filter(filter_str='@')
        )

        self.__dataset_processed[self.__twt_txt] = self.__dataset_processed[self.__twt_txt].apply(
            TweetReviewToPdDataFrame.__gen_strip_words_starting_with_filter(filter_str='#')
        )

        self.__dataset_processed[self.__twt_txt] = self.__dataset_processed[self.__twt_txt].apply(
            TweetReviewToPdDataFrame.__gen_filter_numbers(replace_with=' ')
        )

        self.__dataset_processed[self.__twt_lbl] = self.__dataset_processed[self.__twt_lbl].apply(
            self.__gen_process_label()
        )

    def __train_validate_split(self):
        if self.__dataset_processed is None:
            raise Exception("expected __dataset_processed to be set. Call __preprocess_dataset() " +
                            "before calling the method __train_validate_split()")
        self.__train_ds, self.__validate_ds = train_test_split(self.__dataset_processed,
                                                               test_size=self.__train_validate_split_ratio)

    def __clear_ds_preprocess(self):
        self.__dataset = None
        self.__dataset_processed = None

    def __prepare_ds(self):
        if self.__train_ds is None or self.__validate_ds is None:

            self.__ds_reader.load_dataset()
            self.__dataset = self.__ds_reader.get_dataset()

            self.__preprocess_dataset()

            self.__train_validate_split()

            self.__clear_ds_preprocess()

    def prepare_and_get_train_ds(self):
        self.__prepare_ds()
        return self.__train_ds

    def prepare_and_get_validate_ds(self):
        self.__prepare_ds()
        return self.__validate_ds

    def reset(self):
        self.__train_ds = None
        self.__validate_ds = None


if __name__ == "__main__":
    print(TweetReviewToPdDataFrame._TweetReviewReaderCSV__filter_numbers("hello world a23344b ", replace_with=''))
