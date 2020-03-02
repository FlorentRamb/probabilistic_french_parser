# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:44:35 2020

@author: flo-r
"""

import re

# Read data
with open("data/sequoia-corpus", 'r') as f:
    data = f.read().splitlines()

# ignore functional labels for sparsity issue
for i in range(len(data)):
    data[i] = re.sub(r'-[A-Z]{3}|-[A-Z]{1,}_[A-Z]{1,}', '', data[i])

# build list of tokens
token_list = []
for i in range(len(data)):
    # match words(anything but bracket or whitespace) before one or more closed bracket
    tokens = re.findall(r'([^\)|\s]+)(?=\)+)', data[i])
    # append tokens seperated with whitespace as one string
    token_list.append(' '.join(tokens))

# building train/dev/test sets
input_train = token_list[:int(.8*len(data))]
output_train = data[:int(.8*len(data))]
input_dev = token_list[int(.8*len(data)):int(.9*len(data))]
output_dev = data[int(.8*len(data)):int(.9*len(data))]
input_test = token_list[int(.9*len(data)):]
output_test = data[int(.9*len(data)):]

# Write data
data = [input_train, output_train, input_dev, output_dev,
        input_test, output_test]
files = ["input_train", "output_train", "input_dev", "output_dev",
         "input_test", "output_test"]
for i in range(len(data)):
    with open('data/' + files[i], 'w') as f:
        for item in data[i]:
            f.write("%s\n" % item)