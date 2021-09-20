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

from typing import Optional

from fastapi import FastAPI

from server.serve_model import ModelBuildAndPredict

description = """
Sentiment analysis APP API allows you to query review texts and get an immediate response informing if the query was 
negative in nature or positive in nature. It also returns another filed showing the "confidence level". The confidence 
level is used to describe how certain the model is about the response it has provided. Larger the value, larger the 
level of confidence. 

## Getting started: 

Try typing the following in your browser's address bar : `127.0.0.1:8000/predict?review_qry_string="This sentiment API is the best!"`
NOTE: if you are sending an explicit GET request, you must send this in a valid URI format. The browser you use might 
probably do the conversion for you. The response you will get `{"prediction_label":"positive","prediction_confidence":4.926637649536133}`

The prediction_label always returns either "negative" or "positive". The "prediction_confidence" says the level of 
confidence the model has on its response. 

"""

tags_metadata = [
    {
        "name": "review_qry_string",
        "description": "This is the review query argument of type string.",
    },
]

app = FastAPI(
    title="Sentiment analysis APP",
    description=description,
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
    openapi_tags=tags_metadata,
)

model_build_and_predict = ModelBuildAndPredict("trained_weights/weights_20-9-2021.json")
model_build_and_predict.build_model()


@app.get("/predict", tags=["review_qry_string"], )
def read_item(review_qry_string: str):
    """
        Accepts the argument review_qry_string and computes the sentiment from it.
        \f
        :param review_qry_string: User input string.
    """

    prediction = model_build_and_predict.predict(review_qry_string)
    if prediction[0][0] > 0:
        prediction_label = "positive"
        prediction_confidence = float(prediction[0][0])
    else:
        prediction_label = "negative"
        prediction_confidence = float(prediction[0][0] * -1)

    return {"prediction_label": prediction_label, "prediction_confidence": prediction_confidence}
