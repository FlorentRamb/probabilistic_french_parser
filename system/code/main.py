# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:19:43 2020

@author: flo-r
"""

from CYK import CYK_module
from PYEVALB import scorer, parser


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
    input_test = f.read().splitlines()
with open(test_output_path, 'r') as f:
    output_test = f.read().splitlines()


'''
# Run through test set
tests = []
for i, sentence in enumerate(input_test):
    print(i)
    tests.append(CYK.parse_sentence(sentence))
'''


# Evalb

gold = output_test[0][2:-1]
test = CYK.parse_sentence(input_test[0])[2:-1]

gold_tree = parser.create_from_bracket_string(gold)
test_tree = parser.create_from_bracket_string(test[:-2])
#test_tree = parser.create_from_bracket_string(test[:-2])

result = scorer.Scorer().score_trees(gold_tree, test_tree)

print('Recall =' + str(result.recall))
print('Precision =' + str(result.prec))

'''
gold_path = 'gold_corpus.txt'
test_path = 'test_corpus.txt'
result_path = 'result.txt'

scorer.evalb(gold_path, test_path, result_path)
'''