import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from keras.models import load_model
import spacy


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


def max_index(vector):
    max = vector[0]
    max_ind = 0
    for i in range(1, 3):
        if vector[i] > max:
            max = vector[i]
            max_ind = i
    return max_ind


def aspect_extraction(text, model):
    mapping = load_embeddings('glove.6B.50d.txt')

    sentences = sent_tokenize(text)
    aspects = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        model_input = []
        for token in tokens:
            if token in mapping:
                word_embedding = mapping[token.lower()]
                model_input.append(word_embedding)
            else:
                model_input.append(np.zeros(50))
        model_input = add_padding(model_input, 20)
        model_input = np.array(model_input)

        y = model.predict(np.array([model_input]))

        extracted_aspects = []
        i = 0
        for vec in y[0]:
            if i == len(tokens):
                break
            extracted_aspects.append(max_index(vec))
            i += 1

        sent = []
        print(tokens)
        print(extracted_aspects)
        for i in range(len(tokens)):
            sent.append({'token': tokens[i], 'aspect': extracted_aspects[i]})
        aspects.append(sent)

    return aspects


def order_context(sentence, context):
    ordered = []
    for word in sentence:
        if word['token'] in context['linked_words']:
            ordered.append(word['token'])
    return ordered


def merge_words(words):
    sentence = ''
    for word in words:
        sentence += word + ' '
    return sentence


def extract_relevant_sentence(text, model):
    aspects = aspect_extraction(text, model)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    context = []

    for sentence in aspects:
        for word in sentence:
            if word['aspect'] == 0:
                new_aspect = {'aspect': word['token'], 'linked_words': []}
                for token in doc:
                    print(token.text, [str(child) for child in token.children])
                    children = [str(child) for child in token.children]
                    if word['token'] == token.text:
                        new_aspect['linked_words'].append(str(token.head))
                        for child in children:
                            new_aspect['linked_words'].append(child)
                    elif word['token'] in children:
                        for child in children:
                            new_aspect['linked_words'].append(child)
                new_aspect['linked_words'] = order_context(sentence, new_aspect)
                new_aspect['linked_words'] = merge_words(new_aspect['linked_words'])
                context.append(new_aspect)
    print(context)
    return context


def string2vec(text):
    mapping = load_embeddings('glove.6B.50d.txt')
    vec = []

    for word in text.split():
        if word in mapping:
            word_embedding = mapping[word]
            vec.append(word_embedding)
        else:
            vec.append(np.zeros(50))
    vec = add_padding(vec, 20)
    vec = np.array(vec)
    return vec


def predict(text, model):
    sentence = text
    y = model.predict(np.array([string2vec(sentence)]))

    ind = max_index(y[0])
    if ind == 0:
        return "neutral"
    elif ind == 1:
        return "negative"
    elif ind == 2:
        return "positive"


def extract_sentiment(text, model1, model2):
    extracted = extract_relevant_sentence(text, model1)
    result = []

    for aspect in extracted:
        sentiment = predict(aspect['linked_words'], model2)
        print(aspect['aspect'], sentiment)
        result.append({'aspect': aspect['aspect'], 'sentiment': sentiment})

    return result


# model1 = load_model('models/BiLSTM_Aspect.h5')
# model2 = load_model('models/BiLSTM_Sentiment.h5')
# extract_sentiment("While there is a decent menu it should not take ten minutes to get your drinks and 45 for desert", model1, model2)