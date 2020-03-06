# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:19:43 2020

@author: flo-r
"""

from CYK import CYK_module
import argparse
import time


# Read and parse arguments
arg = argparse.ArgumentParser( description='Basic run script for the Parser' )
arg.add_argument( '--input_file', type=str, required=False, default='../data/input_test', help='Path to tokenized text to parse' )
arg.add_argument( '--output_file', type=str, required=False, default='../output/parsed_test', help='Path to write the output parsed text' )
arg = arg.parse_args()


# Paths to data
bracketed_corpus_path = 'data/output_train'
train_corpus_path = 'data/input_train'
embeddings_path = 'data/polyglot-fr.pkl'
test_input_path = arg.input_file

# Build CYK
CYK = CYK_module(bracketed_corpus_path,
                 train_corpus_path,
                 embeddings_path)

# Grammar description
print('')
print('Number of nonterminals ' + str(len(CYK.PCFG.nonterminals)))
print('')
print('Number of grammar rules ' + str(len(CYK.PCFG.grammar_trees)))

# Read test data
with open(test_input_path, 'r') as f: #arg.input_file
    input_test = f.read().splitlines()

# Run through test set
print('')
print('Parsing input ...')
print('')

t1 = time.time()
tests = []
for i, sentence in enumerate(input_test):
    
    print("parsing sentence " + str(i+1) + " over " + str(len(input_test)))
    out_parsing = CYK.parse_sentence(sentence)
    tests.append(out_parsing)
    with open(arg.output_file, 'a') as f:
        if out_parsing == None:
            f.write("No parsing found" + "\n")
        else:
            f.write(out_parsing + "\n")
t2 = time.time()
print('')
print('Input parsed !')
print('Took {:.2f} minutes to parse'.format((t2 - t1)/60))
