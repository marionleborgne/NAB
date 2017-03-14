# ----------------------------------------------------------------------
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import random
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

from nab.detectors.base import AnomalyDetector

# Global hyper-parameters
sequence_length = 100
random_data_dup = 10  # see dropin() method
epochs = 1
batch_size = 50


def dropin(X, y):
    """
    The name suggests the inverse of dropout, i.e. adding more samples.
    See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings
    -using-recurrent-neural-networks/

    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
    """

    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)


def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean


def prepData(data):
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    result_copy = result[:,:]
    np.random.shuffle(result_copy)  # Shuffle in-place
    X = result_copy[:, :-1]
    y = result_copy[:, -1]
    X, y = dropin(X, y)

    return X, y


def build_model():
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100,
              'output': 1}

    model.add(LSTM(
        input_length=sequence_length - 1,
        input_dim=layers['input'],
        output_dim=layers['hidden1'],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers['hidden2'], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers['hidden3'], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers['output']))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop")
    return model


class LstmDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):

        super(LstmDetector, self).__init__(*args, **kwargs)
        self.data = []
        self.recordCount = 0
        self.model = build_model()
        self.maxMse = 0

    def handleRecord(self, inputData):
        """
        Returns a tuple (anomalyScore) between 0 and 1.
        """
        self.recordCount += 1

        if self.recordCount < self.probationaryPeriod:
            self.data.append(inputData)
            return [0.0]
        elif self.recordCount == self.probationaryPeriod:
            # train LSTM
            X_train, y_train = prepData(self.data)
            self.model.fit(X_train, y_train,
                           batch_size=batch_size,
                           nb_epoch=epochs, validation_split=0.05)
            return [0.0]
        else:
            # predict with LSTM
            X_test, y_test = prepData(self.data)  # TODO: should I chunk it?
            predicted = self.model.predict(X_test)
            predicted = np.reshape(predicted, (predicted.size,))
            mse = (y_test - predicted) ** 2
            if mse > self.maxMse:
                self.maxMse = mse
            if self.maxMse > 0:
                score = mse/ self.maxMse
            else:
                score = mse
            return [score]



