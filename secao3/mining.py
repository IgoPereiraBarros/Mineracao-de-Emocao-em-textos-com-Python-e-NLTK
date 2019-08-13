# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk import FreqDist
from nltk.classify import apply_features
from nltk import NaiveBayesClassifier

#nltk.download()

phrases = [
        ('eu sou admirada por muitos','alegria'),
        ('me sinto completamente amado','alegria'),
        ('amar e maravilhoso','alegria'),
        ('estou me sentindo muito animado novamente','alegria'),
        ('eu estou muito bem hoje','alegria'),
        ('que belo dia para dirigir um carro novo','alegria'),
        ('o dia está muito bonito','alegria'),
        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
        ('o amor e lindo','alegria'),
        ('nossa amizade e amor vai durar para sempre', 'alegria'),
        ('estou amedrontado', 'medo'),
        ('ele esta me ameacando a dias', 'medo'),
        ('isso me deixa apavorada', 'medo'),
        ('este lugar e apavorante', 'medo'),
        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
        ('tome cuidado com o lobisomem', 'medo'),
        ('se eles descobrirem estamos encrencados', 'medo'),
        ('estou tremendo de medo', 'medo'),
        ('eu tenho muito medo dele', 'medo'),
        ('estou com medo do resultado dos meus testes', 'medo')
]

stopwords = stopwords.words('portuguese')
'''
def remove_stopwords(text):
    words = []
    for phrase, emotion in text:
        not_stopwords = [p for p in phrase.split() if p not in stopwords]
        words.append((not_stopwords, emotion))
    return words
'''

# Remove os radicais das palavras
def stemming_(text):
    stemmer = RSLPStemmer()
    stemming = []
    for phrase, emotion in text:
        preprocess = [str(stemmer.stem(p)) for p in phrase.split() if p not in stopwords]
        stemming.append((preprocess, emotion))
    return stemming

words_stemmer = stemming_(phrases)

# Retorna todas as palavras sem os radicais e sem as classes(alegria, medo)
def search_words(phrases):
    words_all = []
    for phrase, emotion in phrases:
        words_all.extend(phrase)
    return words_all

s_words = search_words(words_stemmer)
#print(s_words)

# Retorna a frequência das palavras sem os radicais
def search_frequency(words):
    words = FreqDist(words)
    return words

frequency = search_frequency(s_words)
#print(frequency.most_common(50))

# Retorna as palavras únicas
def search_unique_words(frequency):
    freq = frequency.keys()
    return freq

unique_words = search_unique_words(frequency)
#print(unique_words)

# Verifica se uma palavra existe em um documento
def extract_words(document):
    doc = set(document)
    feature = {}
    for u_word in unique_words:
        feature[u_word] = (u_word in doc)
    return feature

extract = extract_words(['admir', 'med', 'sint'])
#print(extract)

# Retorna todas as palavras do documento, verifica se as palavras passada por parametro tem no documento e informe ao final sua classe(alegria ou medo)
dataset = apply_features(extract_words, words_stemmer)
#print(dataset)

# FAZENDO O MODELO COM NAIVE BAYES

# constroi uma tabela de probabilidade
classifier = NaiveBayesClassifier.train(dataset)
print(classifier.show_most_informative_features(5))


# Testando novos classificador com novos dados
phrase_test = 'Hoje fui roubado, estou apavorado'
test_stemming = []
stemmer = RSLPStemmer()
for word in phrase_test.split():
    w_stemming = [p for p in word.split()]
    test_stemming.append(str(stemmer.stem(w_stemming[0])))
print(test_stemming)

new = extract_words(test_stemming)
print(new)

print(classifier.classify(new))

distribution = classifier.prob_classify(new)
for class_ in distribution.samples():
    print('%s: %f' % (class_, distribution.prob(class_)))