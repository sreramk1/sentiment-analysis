import numpy as np
import tensorflow as tf

from model_train_ds_abst import TrainValidateBase
from model_train_ds_abst import DataSetBase


def default_loss():
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)


def default_optimizer():
    return tf.keras.optimizers.Adam(1e-4)


def default_metrics():
    return ['accuracy']


class TrainSentimentModel(TrainValidateBase):

    def __init__(self,
                 tf_dataset: DataSetBase,
                 model,
                 loss=None,
                 optimizer=None,
                 metrics=None,
                 train_epochs=10,
                 train_validation_steps=30,
                 ):
        if loss is None:
            loss = default_loss()

        if optimizer is None:
            optimizer = default_optimizer()

        if metrics is None:
            metrics = default_metrics()

        self.__tf_dataset = tf_dataset

        self.__model = model

        self.__loss = loss
        self.__optimizer = optimizer
        self.__metrics = metrics

        self.__train_epochs = train_epochs
        self.__train_validation_steps = train_validation_steps

        self.__compiled = False

    def __compile_if_not_already_compiled(self):
        if not self.__compiled:
            self.__model.compile(loss=self.__loss,
                                 optimizer=self.__optimizer,
                                 metrics=self.__metrics)
            self.__compiled = True

    def train(self):
        self.__compile_if_not_already_compiled()
        return self.__model.fit(self.__tf_dataset.prepare_and_get_train_ds(),
                                epochs=self.__train_epochs,
                                validation_data=self.__tf_dataset.prepare_and_get_validate_ds(),
                                validation_steps=self.__train_validation_steps)

    def predict(self, input_data):
        return self.__model.predict(np.array([input_data]))

    def evaluate(self):
        validate_loss, validate_acc = self.__model.evaluate(self.__tf_dataset.prepare_and_get_validate_ds())
        return validate_loss, validate_acc
