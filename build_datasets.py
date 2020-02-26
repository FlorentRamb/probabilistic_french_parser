# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:44:35 2020

@author: flo-r
"""

import re
import os

os.chdir('C:/Users/flo-r/Desktop/Cours MVA/S2/speech_nlp/probabilistic_french_parser')

# Read data
with open("sequoia-corpus+fct.mrg_strict", 'r') as f:
    data = f.read().splitlines()

# ignore functional labels for sparsity issue
for i in range(len(data)):
    data[i] = re.sub(r'-[A-Z]{3}', '', data[i])
    data[i] = re.sub(r'-[A-Z]{2}_[A-Z]{3}', '', data[i])

# build list of tokens
input_CYK = []
for i in range(len(data)):
    input_CYK.append()