# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:07:16 2020

@author: flo-r
"""

import re
import numpy as np
from nltk import Nonterminal
from PCFG import PCFG_module
from OOV import OOV_module


class CYK_module():
    
    
    def __init__(self, bracketed_corpus_path='../data/output_train',
                 train_corpus_path='../data/input_train',
                 embeddings_path='../data/polyglot-fr.pkl'):
        
        print('')
        print('Building PCFG ...')
        self.PCFG = PCFG_module(bracketed_corpus_path)
        print('Done')
        
        print('')
        print('Building OOV module ...')
        self.OOV = OOV_module(train_corpus_path,
                              embeddings_path)
        print('Done')

    
    # Compute probabilistic CYK to recover most likely parsing for a sentence
    # Inspired from https://en.wikipedia.org/wiki/CYK_algorithm#Algorithm
    # Take tokenized input
    def prob_CYK(self, tokens):
    
        n = len(tokens)
        n_symbol = len(self.PCFG.nonterminals)
        self.P = np.zeros((n, n, n_symbol))
        self.back = np.zeros((len(tokens), len(tokens), n_symbol, 3), dtype="int16")
        
        for i in range(n):
            
            candidate = tokens[i]
            
            # Deal with OOV situation
            if not candidate in self.OOV.voc:
                if i > 0:
                    if i < n-1:
                        candidate = self.OOV.oov_word(tokens[i],
                                                      prev_word=tokens[i-1],
                                                      next_word=tokens[i+1])
                    else:
                        candidate = self.OOV.oov_word(tokens[i],
                                                      prev_word=tokens[i-1])
                else:
                    if i < n-1:
                        candidate = self.OOV.oov_word(tokens[i],
                                                      next_word=tokens[i+1])
                    else:
                        candidate = self.OOV.oov_word(tokens[i])
            
            # If no token was found using oov module give uniform proba
            if candidate == None:
                self.P[0,i,:] = 1/n_symbol
            
            else:
                POS_list = self.PCFG.grammar.productions(rhs=candidate)
                for tag in POS_list:
                    self.P[0, i, self.PCFG.nonter2id[tag.lhs()]] = tag.prob()
                    
        for l in range(1, n):
            for s in range(n-l):
                for tree in self.PCFG.grammar_trees:
                    for p in range(l):
                        id_l = self.PCFG.nonter2id[tree[0]]
                        id_r1 = self.PCFG.nonter2id[tree[1]]
                        id_r2 = self.PCFG.nonter2id[tree[2]]
                        
                        p1 = self.P[p, s, id_r1]
                        p2 = self.P[l-p-1, s+p+1, id_r2]
                        p3 = self.P[l, s, id_l]
                        
                        prob_split = tree[3] * p1 * p2
                          
                        if (p1 > 0) & (p2 > 0) & (p3 < prob_split):
                            self.P[l, s, id_l] = prob_split
                            self.back[l, s, id_l, 0] = p
                            self.back[l, s, id_l, 1] = int(id_r1)
                            self.back[l, s, id_l, 2] = int(id_r2)
    
    
    # Build parsing list recursively
    def recursive_parsing(self, l, s, nt_id, tokens):
    
        if l == 0:
            return tokens[s]
    
        else:
            p = self.back[l, s, nt_id, 0]
            nt_left_id = self.back[l, s, nt_id, 1]
            nt_right_id = self.back[l, s, nt_id, 2]
            nt_left = str(self.PCFG.nonterminals[nt_left_id])
            nt_right = str(self.PCFG.nonterminals[nt_right_id])
    
            return [[nt_left, self.recursive_parsing(p, s, nt_left_id, tokens)],
                    [nt_right, self.recursive_parsing(l - p - 1, s + p + 1, nt_right_id, tokens)]]
    
    
    # Transform parsing list to final format
    def list_to_string(self, parsing_list):
    
        if type(parsing_list) == str:
            return parsing_list
    
        else:
            string = ""
            for parsing in parsing_list:
                root = parsing[0]
                parsing_substring = parsing[1]
                string = string + "(" + root + " " + self.list_to_string(parsing_substring) + ")" + " "
            string = string[:-1]
            return string
            
    
    # Parse a sentence to its bracketed form
    def parse_sentence(self, sentence, cor1=True, cor2=True, cor3=True):
    
        tokens = sentence.split()
        n = len(tokens)
    
        if n > 1:
            self.prob_CYK(tokens)
            id_root = self.PCFG.nonter2id[Nonterminal('SENT')]
            if self.P[n-1][0][id_root] == 0: # no valid parsing
                return None
            parsing_list = self.recursive_parsing(n-1, 0, id_root, tokens)
    
        else:
            candidate = tokens[0]
            if candidate not in self.OOV.voc:
                candidate = self.OOV.oov_word(candidate)
            if candidate is None:
                nt = 'NC'
                parsing_list = "(" + nt + " " + tokens[0] + ")"
            else:
                nt = self.PCFG.POS_from_word(candidate)
                parsing_list = "(" + nt + " " + tokens[0] + ")"
        #return(parsing_list)
        
        # Get string from the parsed list
        output = self.list_to_string(parsing_list)
                    
        # Delete created symbols containing '|' and delete closing brackets
        #output = re.sub('\|[^>]+>', '', output)
        if cor1:
            # remove pattern + opening bracket
            output = re.sub(r'([^\s]+)+\|+([^\s]+)', '', output)
            # remove extra white spaces
            output = re.sub(' +', ' ', output)
            count_close = 0
            count_open = 0
            i=0
            #delete extra cloing brackets
            while i < len(output):                
                if output[i] == '(':
                    count_open += 1
                elif output[i] == ')':
                    count_close += 1
                if count_close > count_open:
                    output = output[:i-1] + output[i:]
                    count_close -= 1
                else:
                    i += 1
        
       # Correct merge due to unit rool : (A&&B w) -> (A (B w))
        if cor2:
            count = 0
            for i in range(2, len(output)):
                if i < len(output)-1 and output[i] == '&' and output[i+1] == '&':
                    output = output[:i] + ' (' + output[i+2:]
                    count += 1
                elif output[i] == ')' and count > 0:
                    output = output[:i+1] + count*')' + output[i+1:]
                    count = 0
        
        # Remove artificial 'SENT' tokens with their additive brackets
        if cor3:
            output = output.replace('(SENT ', '')
            count_close = 0
            count_open = 0
            i=0
            #delete extra cloing brackets
            while i < len(output):                
                if output[i] == '(':
                    count_open += 1
                elif output[i] == ')':
                    count_close += 1
                if count_close > count_open:
                    output = output[:i-1] + output[i:]
                    count_close -= 1
                else:
                    i += 1
        
        return "( (SENT " + output + "))"