# -*- coding: utf-8 -*-

#import pandas as pd
#import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk import FreqDist
from nltk.classify import apply_features
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy
from nltk.metrics import ConfusionMatrix
import string

#nltk.download()

def _train_test_split(filename):
    file = open('%s.txt' % filename, 'r')
    train_test_split = {}
    for line in file:
        train_test_split[line.split('    ')[0]] = line.split('    ')[1].rstrip()
    return train_test_split

#train = pd.DataFrame(data={'frase': X_train, 'emoção': y_train})
#test = pd.DataFrame(data={'frase': X_test, 'emoção': y_test})

stopwords = stopwords.words('portuguese')
stopwords.append('vou')
stopwords.append('tão')

# Remove os radicais das palavras
def stemming_(text):
    punc = string.punctuation
    stemmer = RSLPStemmer()
    stemming = []
    for phrase, emotion in text.items():
        nopunc = [str(stemmer.stem(p)) for p in phrase if p not in punc]
        nopunc = ''.join(nopunc)
        preprocess = [str(stemmer.stem(p)) for p in nopunc.split() if p not in stopwords]
        stemming.append((preprocess, emotion))
    return stemming

words_stemmer_train = stemming_(_train_test_split('train'))
words_stemmer_test = stemming_(_train_test_split('test'))

# Retorna todas as palavras sem os radicais e sem as classes(alegria, medo)
def search_words(phrases):
    words_all = []
    for phrase, emotion in phrases:
        words_all.extend(phrase)
    return words_all

s_words_train = search_words(words_stemmer_train)
s_words_test = search_words(words_stemmer_test)
#print(s_words)

# Retorna a frequência das palavras sem os radicais
def search_frequency(words):
    words = FreqDist(words)
    return words

frequency_train = search_frequency(s_words_train)
frequency_test = search_frequency(s_words_test)
#print(frequency.most_common(50))

# Retorna as palavras únicas
def search_unique_words(frequency):
    freq = frequency.keys()
    return freq

unique_words_train = search_unique_words(frequency_train)
unique_words_test = search_unique_words(frequency_test)
#print(unique_words)

# Verifica se uma palavra existe em um documento
def extract_words(document):
    doc = set(document)
    feature = {}
    for u_word in unique_words_train:
        feature[u_word] = (u_word in doc)
    return feature

extract = extract_words(['admir', 'med', 'pesso'])
#print(extract)

# Retorna todas as palavras do documento, verifica se as palavras passada por parametro tem no documento e informe ao final sua classe(alegria ou medo)
dataset_train = apply_features(extract_words, words_stemmer_train)
dataset_test = apply_features(extract_words, words_stemmer_test)
#print(dataset)

# FAZENDO O MODELO COM NAIVE BAYES

# constroi uma tabela de probabilidade
classifier = NaiveBayesClassifier.train(dataset_train)
#print(classifier.labels())
#print(classifier.show_most_informative_features())
#print(accuracy(classifier, dataset_test))

errors = []
for feature, target in dataset_test:
    result = classifier.classify(feature)
    if result != target:
        errors.append((target, result, feature))

for (target, result, feature) in errors:
    print(target, result, feature)
    

# usando a matrix de confução para saber como está os dados em relação de erros e acertos
y_test = []
y_pred = []
for feature, target in dataset_test:
    result = classifier.classify(feature)
    y_test.append(result)
    y_pred.append(target)

cm = ConfusionMatrix(y_test, y_pred)
print(cm)

# 1. Cenário
# 2. Número de classes 16%
# 3. ZeroRules 21,051%

# Testando novos classificador com novos dados
phrase_test = 'eu sinto amor por voce'
test_stemming = []
stemmer = RSLPStemmer()
for word in phrase_test.split():
    w_stemming = [p for p in word.split()]
    test_stemming.append(str(stemmer.stem(w_stemming[0])))
#print(test_stemming)

new = extract_words(test_stemming)
#print(new)

#print(classifier.classify(new))

distribution = classifier.prob_classify(new)
for class_ in distribution.samples():
    #print('%s: %f' % (class_, distribution.prob(class_)))
    pass
