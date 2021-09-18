import tensorflow as tf

from server.model_ds_abst import ModelBuilderBase


class SentimentLSTMDense64Dense32Dense16Dense8Dense1(ModelBuilderBase):

    def __init__(self):
        pass
        # # VOCAB_SIZE = 1000
        # encoder = tf.keras.layers.TextVectorization()
        # # max_tokens=VOCAB_SIZE)
        # encoder.adapt(train_dataset_batched_tf.map(lambda text, label: text))
        #
        # model = tf.keras.Sequential([
        #     encoder,
        #     tf.keras.layers.Embedding(
        #         input_dim=len(encoder.get_vocabulary()),
        #         output_dim=64,
        #         # Use masking to handle the variable sequence lengths
        #         mask_zero=True),
        #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        #
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(32, activation='relu'),
        #     tf.keras.layers.Dense(16, activation='relu'),
        #     tf.keras.layers.Dense(8, activation='relu'),
        #     tf.keras.layers.Dense(1)  # , activation='sigmoid')
        # ])

    def build_model(self):
        pass
