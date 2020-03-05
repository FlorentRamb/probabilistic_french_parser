# Probabilistic French Parser

TD 2 of the course **MVA - Algorithms for Speech and NLP**.


## Description

The task is the folllowing: given a tokenized sentence as input, parse it and output it's most likely bracketed format thanks to the **CYK** algorithm.

The system first learn the **PCFG** on the dataset [sequoia-corpus](https://gforge.inria.fr/frs/?group_id=3597) to learn the grammar and its associated probabilities. 

The system also deals with Out-Of-Vocabulary words using *Levenshtein Distance* and the [polyglot embeddings](https://sites.google.com/site/rmyeid/projects/polyglot) to find close candidate in the training set.


## Example

Input:
`Pourquoi ce thème ?`

Output:
`( (SENT (ADVWH Pourquoi) (NP (DET ce) (NC thème)) (PONCT ?)))`