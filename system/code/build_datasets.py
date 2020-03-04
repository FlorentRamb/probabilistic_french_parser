# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:44:35 2020

@author: flo-r
"""

import re

# Read data
with open("../data/sequoia-corpus", 'r') as f:
    data = f.readlines()

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
data_input = [input_train, input_dev, input_test]
data_output = [output_train, output_dev, output_test]
files_input = ["input_train", "input_dev", "input_test"]
files_output = ["output_train", "output_dev", "output_test"]
for i in range(len(data_input)):
    with open('../data/' + files_input[i], 'w') as f:
        for item in data_input[i]:
            f.write("%s\n" % item)
for i in range(len(data_output)):
    with open('../data/' + files_output[i], 'w') as f:
        for item in data_output[i]:
            f.write(item)