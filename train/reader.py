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


import abc


NotYetLoadedDataFromFileError = Exception(" Dataset was uninitialized." 
                                          "Attempted to retrieve data before loading it.")


class DsReader(abc.ABC):

    @abc.abstractmethod
    def load_dataset(self):
        pass

    @abc.abstractmethod
    def get_dataset(self):
        """
        :raises NotYetLoadedDataFromFileError if it was called before retrieving the data.
        :return:
        """
        pass
