import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import classification_report
from tensorflow_core.python.keras.saving import load_model

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


def one_hot_encoding(category):
    if category == 'B-A':
        return [1, 0, 0]
    elif category == 'I-A':
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def add_padding(vec_sen, vec_cat, max_length):
    if len(vec_sen) < max_length:
        for i in range(max_length - len(vec_sen)):
            vec_sen.append(np.zeros(50))
            vec_cat.append(np.zeros(3))
    else:
        vec_sen = vec_sen[:max_length]
        vec_cat = vec_cat[:max_length]

    return vec_sen, vec_cat


def load_data(file):
    mapping = load_embeddings('glove.6B.50d.txt')
    preprocess = Preprocess()
    file = open(file, "r")
    data_x = []
    data_y = []
    sentence = []
    categories = []
    for line in file:
        if len(line.split()) == 0:
            sentence, categories = add_padding(sentence, categories, 20)
            data_x.append(sentence)
            data_y.append(categories)
            sentence = []
            categories = []
        else:
            if (line.split()[0]).lower() in mapping:
                word_embedding = mapping[(line.split()[0]).lower()]
                word_category = one_hot_encoding(line.split()[2])
                sentence.append(word_embedding)
                categories.append(word_category)
            else:
                sentence.append(np.zeros(50))
                categories.append(one_hot_encoding(line.split()[2]))
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
    for sentence in data:
        tmp = []
        for word_class in sentence:
            tmp.append(class_encoding(word_class))
        result.append(tmp)

    return result


def evaluate_model(model):
    test_x, test_y = load_data("data/Restaurants_poria-test.conll")

    predicted_y = model.predict_classes(test_x)

    test_y = one_hot2class_encoding(test_y)
    predicted_y = flatten(predicted_y)
    test_y = flatten(test_y)

    print(classification_report(test_y, predicted_y, target_names=['B-A', 'I-A', 'O']))


def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(24, return_sequences=True, input_shape=(20, 50))))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model():
    train_x, train_y = load_data("data/Restaurants_poria-train.conll")

    print(train_x.shape)

    model = create_model()
    model.fit(train_x, train_y, epochs=500, batch_size=100)

    evaluate_model(model)

    model.save('models/BiLSTM_Aspect.h5')
    del model


#train_model()

# model = load_model('models/BiLSTM_Aspect.h5')
# evaluate_model(model)