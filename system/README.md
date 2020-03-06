# Options and arguments

**Caution:** you first need to download the the [polyglot embeddings](https://sites.google.com/site/rmyeid/projects/polyglot) for french and to save the file in the `data` folder.

The script `run.sh` will call the script `main.py`in the *\code* folder. It takes as arguments:
1. *input_file*: The path to the input file to be parsed.
2. *output_file*: where will be saved the output of the model.

To use the system execute:
`./run.sh --input_file [input_file] --output_file [output_file]`