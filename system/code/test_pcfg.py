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

a = mod.grammar.productions(rhs=Nonterminal('NC'))
a[-2].rhs()
Out[25]: (NC, NP-ATS|<PP-PONCT>)
'''
