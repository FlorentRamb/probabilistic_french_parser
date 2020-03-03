# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:19:43 2020

@author: flo-r
"""

from CYK import CYK_module

# https://github.com/deloroy/Speech-NLP/tree/master/NLP%20Practical%20-%20Probabilistic%20parsing%20system%20for%20French


# Paths to data
bracketed_corpus_path = '../data/output_train'
train_corpus_path = '../data/input_train'
embeddings_path = '../data/polyglot-fr.pkl'
test_input_path = '../data/input_test'
test_output_path = '../data/output_test'


# Build CYK
CYK = CYK_module(bracketed_corpus_path,
                 train_corpus_path,
                 embeddings_path)


# Read test data
with open(test_input_path, 'r') as f:
    input_test = f.read().lower().splitlines()
with open(test_input_path, 'r') as f:
    input_test = f.read().lower().splitlines()

tests = []
for i, sentence in enumerate(input_test):
    if i % 10 == 0:
        print("Sentence " + str(i) + " over " + str(len(input_test)))
    tests.append(CYK.parse_sentence(sentence))

'''
a = CYK.PCFG.nonterminals
a = [str(elt) for elt in a]
a.sort()
a

from nltk import induce_pcfg, Nonterminal
nt1 = Nonterminal('NP')
nt2 = Nonterminal('ADJWH')
CYK.PCFG.grammar.productions(lhs=nt1, rhs=nt2)
'''