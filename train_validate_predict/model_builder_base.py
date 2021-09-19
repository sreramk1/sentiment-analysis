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
