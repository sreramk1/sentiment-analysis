# sentiment-analysis

Author: Sreram K (sreramk26@gmail.com). All the code in the repository other than the ones in the folder third_party were written by Sreram K. 

## Notebooks:
1. Training from newly initialized weights (randomly initialized) : https://github.com/sreramk1/sentiment-analysis/blob/main/training_notebooks/train_sentiment_p1v1.ipynb
2. Initializing the training with existing weights (read from the json file) : https://github.com/sreramk1/sentiment-analysis/blob/main/training_notebooks/train_sentiment_from_existing_weights_p2v1.ipynb

Please visit those notebooks, and get started immediately with Google Colab. No explicit setting-up is needed to execute those above given notebooks (as long as you execute each cell one after the other). The training API I had written (which I had now shared under Apache License), simplifies the training workflow greatly. It takes not more than 10 seconds to get started. 

## Models I experimented with: 
I primarily chose and experimented with different variants of LSTM (bidirectional) followed by a fully connected layer. 
The final model has a long dense network at the final layer. This improves training performance by a large margin. Adding
additional dense layers with RELU does not cause a hindrance in passing the gradients along the layers. If I had used 
something like a logistic function for each layer of dense neurons, it will cause the gradients to be vanishingly small
as we move towards the LSTM layer. Therefore, the additional RELU layers contributed in adding more degree of freedom in 
updating the weight values without really making it hard for the gradients to pass through each layer. 

## Final metrics from model training and testing

As we can see from the second Notebook: The last training loss was `loss: 0.0184` with the accuracy being 
`accuracy: 0.995`. The measured test-loss that was obtained was: `loss: 0.1755 - accuracy: 0.9545`.

### Results from a few evaluations:
(The query was executed from the Server, by making the HTTP GET calls through the browser)
1. Query URL: `http://127.0.0.1:8000/predict?review_qry_string=%22This%20is%20a%20great%20business!%22`. Query string:
"This is a great business!". Result: `{"prediction_label":"positive","prediction_confidence":2.872779130935669}`. 
HTTP Response header:
```
content-length: 73
content-type: application/json
date: Mon, 20 Sep 2021 15:43:18 GMT
server: uvicorn

Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.5
Connection: keep-alive
DNT: 1
Host: 127.0.0.1:8000
Sec-Fetch-Dest: document
Sec-Fetch-Mode: navigate
Sec-Fetch-Site: none
Sec-Fetch-User: ?1
Sec-GPC: 1
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0

```

3. Query URL: `http://127.0.0.1:8000/predict?review_qry_string=%22This%20is%20the%20worse%20business!%22`. Query string: 
"This is the worse business!". Result: `{"prediction_label":"negative","prediction_confidence":6.614949703216553}`.
HTTP Response header: 
```
content-length: 73
content-type: application/json
date: Mon, 20 Sep 2021 15:46:20 GMT
server: uvicorn

Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.5
Connection: keep-alive
DNT: 1
Host: 127.0.0.1:8000
Sec-Fetch-Dest: document
Sec-Fetch-Mode: navigate
Sec-Fetch-Site: none
Sec-Fetch-User: ?1
Sec-GPC: 1
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0
```