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

from train.reader import DsReader
from train.reader import NotYetLoadedDataFromFileError


class PandasReadDatasetFromCSV(DsReader):

    def __init__(self, csv_path):
        self.__csv_path = csv_path
        self.__dataset = None

    def load_dataset(self):
        self.__dataset = pd.read_csv(self.__csv_path)

    def get_dataset(self):
        if self.__dataset is None:
            raise NotYetLoadedDataFromFileError
        return self.__dataset
