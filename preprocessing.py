import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from autocorrect import spell

snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()


class Preprocess:
    def __int__(self):
        pass

    def autospell(self, text):
        spells = [spell(w) for w in (nltk.word_tokenize(text))]
        return " ".join(spells)

    def to_lower(self, text):
        return text.lower()

    def remove_numbers(self, text):
        output = ''.join(c for c in text if not c.isdigit())
        return output

    def remove_punct(self, text):
        return ''.join(c for c in text if c not in punctuation)

    def remove_Tags(self, text):
        cleaned_text = re.sub('<[^<]+?>', '', text)
        return cleaned_text

    def remove_hashtags(self, text):
        cleaned_text = re.sub('#', '', text)
        return cleaned_text

    def sentence_tokenize(self, text):
        sent_list = []
        for w in nltk.sent_tokenize(text):
            sent_list.append(w)
        return sent_list

    def word_tokenize(self, text):
        return [w for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]

    def remove_stopwords(self, sentence):
        stop_words = stopwords.words('english')
        return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

    def lemmatize(self, text):
        lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for sent in nltk.sent_tokenize(text) for word in
                           nltk.word_tokenize(sent)]
        return " ".join(lemmatized_word)

    def concatenate(self, words):
        text = ''
        for word in words:
            text += word + ' '

        return text

    def preprocess(self, text):
        lower_text = self.to_lower(text)
        sentence_tokens = self.sentence_tokenize(lower_text)
        word_list = []
        for each_sent in sentence_tokens:
            #lemmatizzed_sent = self.lemmatize(each_sent)
            #clean_text = self.remove_numbers(lemmatizzed_sent)
            clean_text = self.remove_punct(each_sent)
            clean_text = self.remove_Tags(clean_text)
            clean_text = self.remove_hashtags(clean_text)
            #clean_text = self.remove_stopwords(clean_text)
            word_tokens = self.word_tokenize(clean_text)
            for i in word_tokens:
                word_list.append(i)
        text = self.concatenate(word_list)
        return text
