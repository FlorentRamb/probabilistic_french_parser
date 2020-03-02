# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:21:50 2020

@author: flo-r
"""

from nltk import induce_pcfg, Nonterminal
from nltk.tree import Tree
from operator import itemgetter
#import os

#os.chdir('C:/Users/flo-r/Desktop/Cours MVA/S2/speech_nlp/probabilistic_french_parser')


class PCFG_module():
    
    
    def __init__(self, bracketed_corpus_path):
                
        # Read bracketed data
        with open(bracketed_corpus_path, 'r') as f:
            self.data = f.read().splitlines()
        
        # Construct trees
        self.grammar = []
        for sentence in self.data:
            tree = Tree.fromstring(sentence[2:-1]) # extract first and last bracket
            tree.collapse_unary() # Eliminate unit rules
            # Eliminate right-hand sides with more than 2 non-terminals
            tree.chomsky_normal_form(horzMarkov=2)
            self.grammar += tree.productions()
        
        # Build PCFG
        self.grammar = induce_pcfg(Nonterminal('SENT'), self.grammar)
        #print(grammar)
        
    
    # Get most likely part-of-speech tag of a word
    def POS_from_word(self, word):
        
        POS_list = self.grammar.productions(rhs=word)
        
        if len(POS_list) == 0:
            print("Word does not exist in grammar")
            return(None)
        
        candidate_list = []
        for candidate in POS_list:
            POS_tag = str(candidate.lhs())
            log_prob = candidate.logprob()
            candidate_list.append((POS_tag, log_prob))
        
        # Sort to get most likely candidate
        candidate_list = sorted(candidate_list, key=itemgetter(1))
        
        return(candidate_list[-1][0])
        
'''
a = grammar.productions(rhs='valider')
print(a)
a = a[0]
print(a)
a.lhs()
a.logprob()
a = grammar.productions(rhs=Nonterminal('NP'))
grammar.productions(lhs=Nonterminal('ADJ'), rhs='chinoise')

a = grammar.productions(rhs='grand')[0]
str(a.lhs())
'''