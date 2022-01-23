**Before doing any activity, we need to activate the sensegram environment using the following command:**
```
conda activate sensegram_env
```

**Train models from a given corpus**
```
python3 create_multiple_models_from_raw_text_file.py /home/sensegram/Desktop/raw_corpus/Ea_L_17547339_W_185687361new.txt -N 300 -n 300 -num_of_models 10
```
Inputs:
Outputs:

**Generate senses from a list of word:**
```
python syed_load_word_and_sense_vectors_for_senses.py 
```
Inputs:
Outputs:

**Reduce the dimensions of word vectors for better visualization.**
```
python3 reduce_dimensions_wordvectors.py 
```
Inputs:
Outputs:

**Bag of words:**
```
python bag_of_words.py
```
Inputs:
Outputs:

# Steps for installation

