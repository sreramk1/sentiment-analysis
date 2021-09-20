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



