# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:37:56 2020

@author: flo-r
"""

import os
import numpy as np

os.chdir('C:/Users/flo-r/Desktop/Cours MVA/S2/speech_nlp/probabilistic_french_parser')

# Read input and output
with open("input_train", 'r') as f:
    input_train = f.read().lower().splitlines()
with open("output_train", 'r') as f:
    output_train = f.read().splitlines()

# Build vocabulary
voc = list(set([word for sentence in input_train for word in sentence.split()]))
voc.sort()
word2id = dict()
for i in range(len(voc)):
    word2id[voc[i]] = i

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

# probabilistic lexicon matrix
prob_lex = np.zeros((len(voc), len(labels)))
#M = np.ones((len(voc), len(labels)))
for tup in couples:
    prob_lex[word2id[tup[1].lower()], label2id[tup[0]]] += 1
prob_lex /= prob_lex.sum(axis=0) # normalize column-wise to get probabilities

# Bigram matrix
bigram_matrix = np.ones((len(voc), len(voc) + 1)) # smoothed matrix
for seq in input_train:
    seq = seq.split()
    bigram_matrix[word2id[seq[0]], 0] += 1 # count each word being first in sequence
    if len(seq) > 1:
        for k in range(1, len(seq)):
          bigram_matrix[word2id[seq[k]], word2id[seq[k-1]] + 1] += 1
bigram_matrix /= bigram_matrix.sum(axis=0) # normalize column-wise to get probabilities
