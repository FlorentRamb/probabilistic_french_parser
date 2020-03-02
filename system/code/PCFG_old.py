# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:37:56 2020

@author: flo-r
"""

import os
import numpy as np

os.chdir('C:/Users/flo-r/Desktop/Cours MVA/S2/speech_nlp/probabilistic_french_parser')

# Read input and output
with open("data/input_train", 'r') as f:
    input_train = f.read().lower().splitlines()
with open("data/output_train", 'r') as f:
    output_train = f.read().splitlines()

# Build vocabulary
voc = list(set([word for sentence in input_train for word in sentence.split()]))
voc.sort()
word2id = dict()
for i in range(len(voc)):
    word2id[voc[i]] = i

# Language model
words_prob = np.zeros(len(voc)) # words occurence probability
bigram_matrix = np.ones((len(voc), len(voc) + 1)) # smoothed bigram matrix
for seq in input_train:
    seq = seq.split()
    words_prob[word2id[seq[0]]] += 1
    bigram_matrix[word2id[seq[0]], 0] += 1 # case when word is first in sequence
    if len(seq) > 1:
        for k in range(1, len(seq)):
            words_prob[word2id[seq[k]]] += 1
            bigram_matrix[word2id[seq[k]], word2id[seq[k-1]] + 1] += 1
bigram_matrix /= bigram_matrix.sum(axis=0) # normalize column-wise to get probabilities
words_prob /= words_prob.sum() # normalize to get probabilities

# couples label/word
couples = []
for sentence in output_train:
    sentence = sentence.split()
    for i in range(len(sentence)-1):
        if sentence[i][0] == '(' and sentence[i+1][-1] == ')':
            couples.append((sentence[i].replace('(', ''), sentence[i+1].replace(')', '')))

# Extract labels
labels = list(set([tup[0] for tup in couples]))
labels.sort()
label2id = dict()
for i in range(len(labels)):
    label2id[labels[i]] = i

# probabilistic lexicon
labels_prob = np.zeros(len(labels))
lexicon_prob = np.zeros((len(voc), len(labels)))
#M = np.ones((len(voc), len(labels)))
for tup in couples:
    labels_prob[label2id[tup[0]]] += 1
    lexicon_prob[word2id[tup[1].lower()], label2id[tup[0]]] += 1
lexicon_prob /= lexicon_prob.sum(axis=0) # normalize column-wise to get probabilities
labels_prob /= labels_prob.sum() # normalize to get probabilities