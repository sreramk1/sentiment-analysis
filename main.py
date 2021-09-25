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

from fastapi import FastAPI

from server.sentiment_v1.server_builder import ServerBuilder

app = FastAPI(
    title="Sentiment analysis APP",
    description=ServerBuilder.get_description(),
    version="1.0.0",
    contact={
        "name": "Sreram K",
        "url": "https://www.linkedin.com/in/k-sreram-a04a90b7/",
        "email": "sreramk26@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_tags=ServerBuilder.get_tags_metadata(),
)

server_builder = ServerBuilder(app)
server_builder.build_server()
