**Before doing any activity, we need to activate the sensegram environment using the following command:**
```
conda activate sensegram_env
```

# Train models from a given corpus
## Command:
```
python3 create_multiple_models_from_raw_text_file.py /home/sensegram/Desktop/raw_corpus/Ea_L_17547339_W_185687361new.txt -N 300 -n 300 -num_of_models 10
```
## Inputs:
 - train_corpus [Path to a training corpus in text form]. NO FLAG NEEDED.
 - -cbow [Use the continuous bag of words model (default is 1, use 0 for the skip-gram model)]. FLAG NEEDED
 - -size [Set size of word vectors (default is 300)]. FLAG NEEDED
 - -N [Number of nodes in each ego-network (default is 200)]. FLAG NEEDED
 - -n [Maximum number of edges a node can have in the network (default is 200)]. FLAG NEEDED. (Does not effect the results, but use N and n equal to keep consistent filenames.)
 - -num_of_models [Number of models to be trained (default 1)]. FLAG NEEDED.
 - -output_models_base_path [output_models_base_path (default model/). This is the where the models will be stored. Recommended to use the default path]. FLAG NEEDED.

## Outputs:
Trains the models and store them under _output_models_base_path/corpus_name_


# Generate senses from a list of word:
## Command:
```
python syed_load_word_and_sense_vectors_for_senses.py 
```
## Inputs:
Set the following variables in the file before executing the command.
 - corpus_name 
 - model_base_dir [Directory in which the models are present]

Effective directory path used to load the models is: 
_model_base_dir/corpus_name_

 - The code asks for the path of file which contains words to get senses.
## Outputs:
It creates a directory _word_senses/corpus_name_ which contains csv files for each word provided in the input.


# Reduce the dimensions of word vectors for better visualization using PCA.
## Command:
```
python3 reduce_dimensions_wordvectors.py 
```
## Inputs:
 - corpus_name [Name of the corpus. Models of which will be used to generate word vectors. Models should be stored in the directory model/]
 - words_to_get_senses_filename [Path to the file which contains the list of words. Each line should have only one word.]
 - path_to_save_file [Line 144-147. Path where to save the PCA output file.]
## Outputs:
 - Writes a CSV file on the specified path. It has columns: word, X,	Y,	Z,	word_vector


# Bag of words:
## Command:
```
python bag_of_words.py
```
## Inputs:
## Outputs:

# Steps for installation

