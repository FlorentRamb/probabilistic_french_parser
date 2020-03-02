# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:04:38 2020

@author: flo-r
"""

import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
from PCFG import PCFG_module

# https://github.com/deloroy/Speech-NLP/tree/master/NLP%20Practical%20-%20Probabilistic%20parsing%20system%20for%20French


class OOV_module():
    
    
    def __init__(self,
                 train_corpus_path='../data/input_train',
                 bracketed_corpus_path='../data/sequoia-corpus',
                 embeddings_path='../data/polyglot-fr.pkl'):
        
        # Load polyglot embeddings
        self.words, self.embeddings = pickle.load(open(
            embeddings_path, 'rb'
            ), encoding='latin1')
        self.wordemb2id = {w:i for (i, w) in enumerate(self.words)}
        
        # Read training corpus
        with open(train_corpus_path, 'r') as f:
            self.input_train = f.read().lower().splitlines()
        
        # Build vocabulary
        self.voc = list(set([word for sentence in self.input_train for word in sentence.split()]))
        self.voc.sort()
        self.word2id = {w:i for (i, w) in enumerate(self.voc)}
        
        # Language model
        self.words_prob = np.zeros(len(self.voc)) # words occurence probability
        self.bigram_matrix = np.ones((len(self.voc), len(self.voc) + 1)) # smoothed bigram matrix
        for seq in self.input_train:
            seq = seq.split()
            self.words_prob[self.word2id[seq[0]]] += 1
            self.bigram_matrix[self.word2id[seq[0]], 0] += 1 # case when word is first in sequence
            if len(seq) > 1:
                for k in range(1, len(seq)):
                    self.words_prob[self.word2id[seq[k]]] += 1
                    self.bigram_matrix[self.word2id[seq[k]],
                                       self.word2id[seq[k-1]] + 1] += 1
        # normalize column-wise to get probabilities
        self.bigram_matrix /= self.bigram_matrix.sum(axis=0)
        self.bigram_matrix = np.log(self.bigram_matrix)
        # normalize to get probabilities
        self.words_prob /= self.words_prob.sum()
        self.words_prob = np.log(self.words_prob)
        
        # Get embeddings of words in the train set
        self.train_word_emb = dict()
        for w in self.voc:
            if w in self.words:
                self.train_word_emb[w] = self.embeddings[self.wordemb2id[w]]
        
        # Build PCFG
        self.PCFG = PCFG_module(bracketed_corpus_path)


    # Levenshtein distance between two strings
    def lev_dist(self, s1, s2):
        m = np.zeros((len(s1)+1, len(s2)+1))
        for i in range(1, len(s1)+1):
            m[i,0] = i
        for i in range(1, len(s2)+1):
            m[0,i] = i
        for i in range(1, len(s1)+1):
            for j in range(1, len(s2)+1):
                if s1[i-1] == s2[j-1]:
                    m[i,j] = min(m[i-1,j]+1, m[i,j-1]+1, m[i-1,j-1])
                else:
                    m[i,j] = min(m[i-1,j]+1, m[i,j-1]+1, m[i-1,j-1]+1)
        return(m[-1, -1])


    # Assign Part-of-Speech to out of vocabulary word
    # Use a mix of embeddings similarity, Levenshtein distance and language model
    def oov_word(self, word, prev_word=None, next_word=None):
        
        # Find most similar words in train wrt Levenshtein distance
        words_lv = {1:[], 2:[]}
        for w in self.voc:
            distance = self.lev_dist(word, w)
            if distance == 2:
                words_lv[2].append(w)
            if distance == 1:
                words_lv[1].append(w)
        
        # First consider candidates with distance 1
        if len(words_lv[1]) > 0:
            
            # If unique candidate return it
            if len(words_lv[1]) == 1:
                return(self.PCFG.POS_from_word(words_lv[1][0]))
            
            # If more than one candidate use language probabilities
            candidate_list = []
            for candidate in words_lv[1]:
                log_p = 0
                if prev_word != None and prev_word in self.voc:
                    log_p += self.words_prob[self.word2id[prev_word]] + \
                        self.bigram_matrix[self.word2id[candidate], self.word2id[prev_word]+1]
                else:
                    log_p += self.words_prob[self.word2id[candidate]]
                if next_word != None and next_word in self.voc:
                    log_p += self.bigram_matrix[self.word2id[next_word], self.word2id[candidate]+1]
                candidate_list.append((candidate, log_p))
            
            # Sort to get most likely candidates
            candidate_list = sorted(candidate_list, key=itemgetter(1))
            return(self.PCFG.POS_from_word(candidate_list[-1][0]))
        
        # Then consider candidates with distance 2
        if len(words_lv[2]) > 0:
            
            # If unique candidate return it
            if len(words_lv[2]) == 1:
                return(self.PCFG.POS_from_word(words_lv[2][0]))
            
            # If more than one candidate use language probabilities
            candidate_list = []
            for candidate in words_lv[2]:
                log_p = 0
                if prev_word != None and prev_word in self.voc:
                    log_p += self.words_prob[self.word2id[prev_word]] + \
                        self.bigram_matrix[self.word2id[candidate], self.word2id[prev_word]+1]
                else:
                    log_p += self.words_prob[self.word2id[candidate]]
                if next_word != None and next_word in self.voc:
                    log_p += self.bigram_matrix[self.word2id[next_word], self.word2id[candidate]+1]
                candidate_list.append((candidate, log_p))
            
            # Sort to get most likely candidate
            candidate_list = sorted(candidate_list, key=itemgetter(1))
            return(self.PCFG.POS_from_word(candidate_list[-1][0]))
        
        # If no word in train with levenshtein distance below 2
        # get most similar words in train wrt embeddings
        if word in self.words:
            word_emb = self.embeddings[self.wordemb2id[word]]
            candidate_list = []
            for w in self.train_word_emb.keys():
                sim = cosine_similarity(word_emb.reshape((1, -1)),
                                        self.train_word_emb[w].reshape((1, -1)))
                candidate_list.append((w, sim))
            candidate_list = sorted(candidate_list, key=itemgetter(1))
            return(self.PCFG.POS_from_word(candidate_list[-1][0]))
        
        # If none of these methods is successful return None
        print("OOV failure:")
        print("Did not find any close word")
        return(None)
