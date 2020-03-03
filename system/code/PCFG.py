# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:21:50 2020

@author: flo-r
"""

from nltk import induce_pcfg, Nonterminal
from nltk.tree import Tree
from operator import itemgetter


class PCFG_module():
    
    
    def __init__(self, bracketed_corpus_path='../data/output_train'):
                
        # Read bracketed data
        with open(bracketed_corpus_path, 'r') as f:
            self.data = f.read().splitlines()
        
        # Construct trees with CNF
        self.grammar = []
        for sentence in self.data:
            
            # delete first and last bracket
            tree = Tree.fromstring(sentence[2:-1])
            
            # Eliminate right-hand sides with more than 2 non-terminals
            tree.chomsky_normal_form(horzMarkov=1)
            
            # Eliminate unit rules
            tree.collapse_unary(collapsePOS=True,
                                collapseRoot=True,
                                joinChar='&&')
            
            self.grammar += tree.productions()
        
        # Build PCFG
        self.grammar = induce_pcfg(Nonterminal('SENT'), self.grammar)
        
        # Extract all non terminals and grammar trees
        self.nonterminals = []
        self.grammar_trees = []
        for production in self.grammar.productions():
            lhs = production.lhs()
            rhs = production.rhs()
            if len(rhs) == 2:
                if type(rhs[0]) != str:
                    self.grammar_trees.append([lhs, rhs[0], rhs[1], production.prob()])
            self.nonterminals.append(lhs)
        self.nonterminals = list(set(self.nonterminals))
        self.nonter2id = {nt:i for (i, nt) in enumerate(self.nonterminals)}
        
    
    # Get most likely part-of-speech tag of a word
    def POS_from_word(self, word):
        
        POS_list = self.grammar.productions(rhs=word)
        
        if len(POS_list) == 0:
            print("Word [" + word + "] does not exist in grammar")
            return(None)
        
        candidate_list = []
        for candidate in POS_list:
            POS_tag = str(candidate.lhs())
            log_prob = candidate.logprob()
            candidate_list.append((POS_tag, log_prob))
        
        # Sort to get most likely candidate
        candidate_list = sorted(candidate_list, key=itemgetter(1))
        
        return(candidate_list[-1][0])
