# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:54:52 2020

@author: flo-r
"""

from PYEVALB import scorer, parser
import matplotlib.pyplot as plt
import numpy as np


# Paths to data
parsed_output_path = '../output/parsed_test'
test_input_path = '../data/input_test'
test_output_path = '../data/output_test'

# Read data
with open(parsed_output_path, 'r') as f:
    parsed_output = f.read().splitlines()
with open(test_input_path, 'r') as f:
    test_input = f.read().splitlines()
with open(test_output_path, 'r') as f:
    test_output = f.read().splitlines()

# Compute metrics
precisions = []
recalls = []
lengths = []
failures = 0
bugs = 0
for gold, test, sent in zip(test_output, parsed_output, test_input):
    if test == 'No parsing found':
        failures += 1
    else:
        try:
            gold_tree = parser.create_from_bracket_string(gold[2:-1])
            test_tree = parser.create_from_bracket_string(test[2:-1])
            result = scorer.Scorer().score_trees(gold_tree, test_tree)
            
            len_sentence = len(sent)
            lengths.append(len_sentence)
            print('')
            print('Sentence length: ' + str(len(gold)))
            print('Recall =' + str(result.recall))
            print('Precision =' + str(result.prec))
            recalls.append(result.recall)
            precisions.append(result.prec)
        except:
            bugs +=1

print('')
print('Parsing failures for ' + str(failures + bugs) + 'sentences')

print('')
print('Average precision: ' + str(np.mean(precisions)))

print('')
print('Average recall: ' + str(np.mean(recalls)))


# Plots

plt.scatter(lengths, precisions)
plt.grid()
plt.title('Precision VS sentence length')
plt.xlabel('number of tokens')
plt.ylabel('precision')
plt.show()

plt.scatter(lengths, recalls)
plt.grid()
plt.title('Recall VS sentence length')
plt.xlabel('number of tokens')
plt.ylabel('recall')
plt.show()
