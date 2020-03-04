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
                                    
# test OOV
                                    
# Assign Part-of-Speech to out of vocabulary word
    # Use a mix of embeddings similarity, Levenshtein distance and language model
    def oov_word(self, word, prev_word=None, next_word=None, tresh_lev_dist=2):
        
        # List of candidate among vocabulary
        candidate_list = []
        
        # Find most similar words in train wrt Levenshtein distance
        for w in self.voc:
            distance = self.lev_dist(word, w)
            if distance <= tresh_lev_dist:
                candidate_list.append(w)
        
        # Get top 5 most similar words in train wrt embeddings
        if word in self.words:
            word_emb = self.embeddings[self.wordemb2id[word]]
            candidate_list_emb = []
            for w in self.train_word_emb.keys():
                sim = cosine_similarity(word_emb.reshape((1, -1)),
                                        self.train_word_emb[w].reshape((1, -1)))
                candidate_list_emb.append((w, sim))
            candidate_list_emb = sorted(candidate_list_emb, key=itemgetter(1))
            candidate_list += [elt[0] for elt in candidate_list_emb[-5:]]
        
        # If none of these methods is successful return None
        if len(candidate_list) == 0:
            print("OOV failure: did not find any close word for [" + word + "]")
            return(None)
        
        if len(candidate_list) == 1:
            return(candidate_list[0])

        # If more than one candidate use language probabilities
        # We need to deal with 4 cases depending on previous and next words
        if prev_word == None and next_word == None:
            for candidate in candidate_list:
                log_p = self.words_prob[self.word2id[candidate]]
                candidate_list.append((candidate, log_p))
                    
        elif prev_word != None and prev_word in self.voc and next_word == None:
            for candidate in candidate_list:
                log_p = self.bigram_matrix[self.word2id[candidate],
                                           self.word2id[prev_word]]
                candidate_list.append((candidate, log_p))
                    
        elif prev_word == None and next_word != None and next_word in self.voc:
            for candidate in candidate_list:
                log_p = self.bigram_matrix[self.word2id[next_word],
                                           self.word2id[candidate]] + \
                    self.words_prob[self.word2id[candidate]]
                candidate_list.append((candidate, log_p))
        
        elif prev_word != None and prev_word in self.voc and next_word != None and next_word in self.voc:
            for candidate in candidate_list:
                log_p = self.bigram_matrix[self.word2id[candidate],
                                           self.word2id[prev_word]] + \
                    self.bigram_matrix[self.word2id[next_word],
                                       self.word2id[candidate]]
                candidate_list.append((candidate, log_p))
        
        # Sort to get most likely candidate
        candidate_list = sorted(candidate_list, key=itemgetter(1))
        return(candidate_list[-1][0])
        
import re
#output = '(SENT (NP w1) (Ssub|<PONCT> (PONCT w2)))'
output = CYK.parse_sentence(input_test[0], cor1=False, cor2=False, cor3=False)
output = output[8:-2]
print(output)
output = re.sub(r'([^\s]+)+\|+([^\s]+)', '', output)
# remove extra white spaces
output = re.sub(' +', ' ', output)
print(output)
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

'''