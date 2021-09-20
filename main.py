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

app = FastAPI()

model_build_and_predict = ModelBuildAndPredict("trained_weights/weights_20-9-2021.json")
model_build_and_predict.build_model()


@app.get("/predict")
def read_item(review_qry_string: Optional[str] = None):
    """
        Accepts the argument review_qry_string and computes the sentiment from it.
    """

    prediction = model_build_and_predict.predict(review_qry_string)
    if prediction[0][0] > 0:
        prediction_label = "positive"
        prediction_confidence = float(prediction[0][0])
    else:
        prediction_label = "negative"
        prediction_confidence = float(prediction[0][0] * -1)

    return {"prediction_label": prediction_label, "prediction_confidence": prediction_confidence}
