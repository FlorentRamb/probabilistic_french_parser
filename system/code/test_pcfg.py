# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:37:30 2020

@author: flo-r
"""

from nltk import induce_pcfg, Nonterminal
from nltk.tree import Tree

bracketed_corpus_path='../data/output_train'

# Read bracketed data
with open(bracketed_corpus_path, 'r') as f:
    data = f.read().splitlines()

print(data[21])


# Construct trees
tree = Tree.fromstring(data[21][2:-1])
productions1 = tree.productions()

nonterminals1 = []
for p in productions1:
    nonterminals1.append(p.lhs())
nonterminals1 = list(set(nonterminals1))


# Construct trees
tree = Tree.fromstring(data[21][2:-1])
tree.collapse_unary()
tree.chomsky_normal_form(horzMarkov=2)
productions2 = tree.productions()

nonterminals2 = []
for p in productions2:
    nonterminals2.append(p.lhs())
nonterminals2 = list(set(nonterminals2))


# Construct trees
tree = Tree.fromstring(data[21][2:-1])
tree.collapse_unary(collapsePOS=True, joinChar='&&')
tree.chomsky_normal_form(horzMarkov=2)
productions3 = tree.productions()

nonterminals3 = []
for p in productions3:
    nonterminals3.append(p.lhs())
nonterminals3 = list(set(nonterminals3))


# Build PCFG
grammar = induce_pcfg(Nonterminal('SENT'), productions3)
#print(grammar)

'''
# Old CYK
        for l in range(1, n):
            for s in range(n-l): #range(n-l+1)
                for p in range(l): #range(l-1)
                    for nt1 in self.PCFG.nonterminals:
                        for nt2 in self.PCFG.nonterminals:
                        
                            prod_list = self.PCFG.grammar.productions(lhs=nt1, rhs=nt2)
                            
                            if len(prod_list) == 0:
                                continue
                                
                            for prod in prod_list:
                                nt3 = prod.rhs()[1]
                                p1 = prod.prob()
                                p2 = self.P[p, s, self.PCFG.nonter2id[nt2]]
                                p3 = self.P[l-p, s+p, self.PCFG.nonter2id[nt3]]
                                p4 = self.P[l, s, self.PCFG.nonter2id[nt1]]
                                prob_split = p1 * p2 * p3
                                
                                if (p2 > 0) & (p3 > 0) & (p4 < prob_split):
                                    print("in if")
                                    self.P[l, s, self.PCFG.nonter2id[nt1]] = prob_split
                                    self.back[l, s, self.PCFG.nonter2id[nt1], 0] = p
                                    self.back[l, s, self.PCFG.nonter2id[nt1], 1] = self.PCFG.nonter2id[nt2]
                                    self.back[l, s, self.PCFG.nonter2id[nt1], 2] = self.PCFG.nonter2id[nt3]
'''