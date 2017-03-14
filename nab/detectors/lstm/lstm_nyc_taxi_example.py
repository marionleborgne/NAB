""" Inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
Uses the TensorFlow backend
The basic idea is to detect anomalies in a time-series.
"""
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from numpy import arange, sin, pi, random

np.random.seed(1234)

# Global hyper-parameters
sequence_length = 100
random_data_dup = 10  # See dropin() function
epochs = 1
batch_size = 50
READ_CSV = True


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
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)


def gen_wave():
    """ Generate a synthetic wave by adding up a few sine waves and some noise
    :return: the final wave
    """
    t = np.arange(0.0, 10.0, 0.01)
    wave1 = sin(2 * 2 * pi * t)
    noise = random.normal(0, 0.1, len(t))
    wave1 = wave1 + noise
    print("wave1", len(wave1))
    wave2 = sin(2 * pi * t)
    print("wave2", len(wave2))
    t_rider = arange(0.0, 0.5, 0.01)
    wave3 = sin(10 * pi * t_rider)
    print("wave3", len(wave3))
    insert = int(0.8 * len(t))
    wave1[insert:insert + 50] = wave1[insert:insert + 50] + wave3
    return wave1 + wave2


def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean


def read_csv(csv_path):
    print csv_path
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df.value.values.astype("float64")


def get_split_prep_data(train_start, train_end,
                        test_start, test_end):
    if READ_CSV:
        data = read_csv('../../../data/realKnownCause/nyc_taxi.csv')
    else:
        data = gen_wave()
    print("Length of Data", len(data))

    if test_end is None:
        test_end = len(data)
    # train data
    print "Creating train data..."

    result = []
    for index in range(train_start, train_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    print "Mean of train data : ", result_mean
    print "Train data shape  : ", result.shape

    train = result[train_start:train_end, :]
    np.random.shuffle(train)  # shuffles in-place
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_train, y_train = dropin(X_train, y_train)

    # test data
    print "Creating test data..."

    result = []
    for index in range(test_start, test_end - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)  # shape (samples, sequence_length)
    result, result_mean = z_norm(result)

    print "Mean of test data : ", result_mean
    print "Test data shape  : ", result.shape

    X_test = result[:, :-1]
    y_test = result[:, -1]

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test


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

    model.add(LSTM(
        layers['hidden2'],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers['hidden3'],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers['output']))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start
    return model


def run_network():
    global_start_time = time.time()

    print 'Loading data... '
    X_train, y_train, X_test, y_test = get_split_prep_data(0, 400, 400, None)
    print '\nData Loaded. Compiling...\n'
    model = build_model()

    # Train
    print("Training...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
              validation_split=0.05)

    # Predict
    print("Predicting...")
    predicted = model.predict(X_test)
    print("Reshaping predicted")
    predicted = np.reshape(predicted, (predicted.size,))

    # Plot
    plt.figure(1)
    ax1 = plt.subplot(311)
    ax1.set_title("Actual Signal")
    ax1.plot(y_test[:len(y_test)], 'b')

    ax2 = plt.subplot(312, sharex=ax1)
    ax2.set_title("Predicted Signal")
    ax2.plot(predicted[:len(y_test)], 'g')

    mse = ((y_test - predicted) ** 2)
    ax3 = plt.subplot(313, sharex=ax2)
    ax3.set_title("Squared Error")
    ax3.plot(mse, 'r')

    plt.show()

    print 'Training duration (s) : ', time.time() - global_start_time

    return model, y_test, predicted


run_network()
