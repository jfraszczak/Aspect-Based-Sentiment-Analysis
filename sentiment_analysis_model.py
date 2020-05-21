from random import random

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import classification_report
from tensorflow_core.python.keras.models import load_model

from preprocessing import Preprocess


def load_embeddings(path):
    mapping = dict()

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            splitted = line.split(" ")
            mapping[splitted[0]] = np.array(splitted[1:], dtype=float)
    return mapping


def add_padding(vector, max_length):
    if len(vector) < max_length:
        for i in range(max_length - len(vector)):
            vector.append(np.zeros(50))
    else:
        vector = vector[:max_length]
    return vector


def one_hot_encoding(sentiment):
    if sentiment == 2:
        return [1, 0, 0]
    elif sentiment == 0:
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def load_data(file):
    mapping = load_embeddings('glove.6B.50d.txt')
    preprocess = Preprocess()
    data = pd.read_csv(file, encoding='latin-1', names=['sentiment', 'id', 'date', 'q', 'nick', 'tweet'])
    data = data.sample(frac=1)
    data = data[:100000]

    data_x = []
    data_y = []
    for index in data.index:
        row = data.loc[index, :]
        if row['sentiment'] != 2:
            row['tweet'] = preprocess.preprocess(row['tweet'])
            tweet = []
            for word in row['tweet'].split():
                if word in mapping:
                    word_embedding = mapping[word]
                    tweet.append(word_embedding)
                else:
                    tweet.append(np.zeros(50))
            tweet = add_padding(tweet, 20)
            data_x.append(tweet)
            data_y.append(one_hot_encoding(row['sentiment']))
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    return data_x, data_y


def class_encoding(category):
    if np.array_equal(np.array([1., 0., 0.]), category):
        return 0
    elif np.array_equal(np.array([0., 1., 0.]), category):
        return 1
    else:
        return 2


def flatten(array):
    result = []
    for a in array:
        for v in a:
            result.append(v)
    return result


def one_hot2class_encoding(data):
    result = []
    for y in data:
        result.append(class_encoding(y))

    return result


def evaluate_model(model):
    test_x, test_y = load_data("data/testdata.manual.2009.06.14.csv")

    predicted_y = model.predict_classes(test_x)
    test_y = one_hot2class_encoding(test_y)

    print(classification_report(test_y, predicted_y, target_names=['negative', 'positive']))


def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(24, input_shape=(20, 50))))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model():
    train_x, train_y = load_data("data/training.1600000.processed.noemoticon.csv")

    model = create_model()
    model.fit(train_x, train_y, epochs=500, batch_size=100)

    evaluate_model(model)
    model.save('models/BiLSTM_Sentiment.h5')
    del model


def retrain_model(file):
    train_x, train_y = load_data("data/training.1600000.processed.noemoticon.csv")

    model = load_model(file)
    model.fit(train_x, train_y, epochs=1000, batch_size=100)

    evaluate_model(model)
    model.save(file)
    del model


#train_model()

# model = load_model('models/BiLSTM_Sentiment.h5')
# evaluate_model(model)