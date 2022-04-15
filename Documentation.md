# SENSE2VEC
https://github.com/explosion/sense2vec

## Create environment
 - conda create --name sense2vec
 - conda activate sense2vec
 - conda install matplotlib
 - pip install sense2vec
 - python3 -m spacy download en_core_web_sm

## Usage 
 - python standalone_usage.py
 - python usage_as_a_spaCy_pipeline_component.py

# SENSEGRAM
**Before doing any activity, we need to activate the sensegram environment using the following command:**
```
conda activate sensegram_env
```

The reason for doing this is to isolate the working enivronment of the sensegram from the other projects. 
Otherwise, there is a risk that some other project requires a package which conflicts with the packages required for sensegram.
While working on other projects, make sure that the **sensegram_env** is not activated. 

# Train models from a given corpus

```
usage: 1_create_multiple_models_from_raw_text_file.py [-h] [-cbow CBOW] [-size SIZE] [-N N] [-n N] [-num_of_models NUM_OF_MODELS] [-output_models_base_path OUTPUT_MODELS_BASE_PATH] train_corpus
```
## Command:
```
python3 1_create_multiple_models_from_raw_text_file.py /home/sensegram/Desktop/raw_corpus/Ea_L_17547339_W_185687361new.txt -cbow 1 -size 300 -N 300 -n 300 -num_of_models 10 -output_models_base_path model/```
## Inputs:
 - train_corpus: FLAG NEEDED. Path to a training corpus in text form. The sensgram models will be trained on this corpus. NO FLAG NEEDED. 
 - -cbow: Select the type of Word2Vec algorithm for training the model. Use 1 for the continuous bag of words model, use 0 for the skip-gram model (default is 1). The CBOW model learns the embedding by predicting the current word based on its context. The continuous skip-gram model learns by predicting the surrounding words given a current word. The continuous skip-gram model learns by predicting the surrounding words given a current word.
     - CBOW converges faster than Skip-gram. 
     - CBOW learn better syntactic relationships between words while Skip-gram is better in capturing better semantic relationships. In practice, this means that for the word 'cat' CBOW would retrive as closest vectors morphologically similar words like plurals, i.e. 'cats' while Skip-gram would consider morphologically different words (but semantically relevant) like 'dog' much closer to 'cat' in comparison.  
     - Because Skip-gram rely on single words input, it is less sensitive to overfit frequent words, because even if frequent words are presented more times that rare words during training, they still appear individually, while CBOW is prone to overfit frequent words because they appear several time along with the same context.
     - Skip-gram works well with a small amount of the training data, represents well even rare words or phrases, whereas CBOW gives slightly better accuracy for the frequent words.
     <img src="https://user-images.githubusercontent.com/22868291/162568127-e946366c-d640-4502-bc6a-8bc747f175bb.png" alt="cbow skipgram diagram" width="500"/>
     
     - Example of skip-gram
     <img src="https://user-images.githubusercontent.com/22868291/163095522-bb3ebccf-d27a-4ebf-ae3d-f77f15d5d5b4.png" alt="skipgram diagram" width="500"/>

     - Example of CBOW
     <img src="https://user-images.githubusercontent.com/22868291/163096285-cced378d-a7bb-4cc0-8442-60dfdde3e36e.png" alt="cbow diagram" width="500"/>

- -size: Size of the vector of each word present in the sensegram model. Set size of word vectors (default is 300). A word embedding/vector is a learned representation for text where words that have the same meaning have a similar representation. The choice of dimensions is very much dependent on corpus size. A reasonable approach, for any corpus with unique word count greater than a million, is to start with 300. Typically, the length of these vectors 100–300 dimensions but it is recommended to use 300 as per the academic literature. Pros of smaller dimensionalities is that it requires less training time but it will capture less information. In contrast larger dimensionalities will capture more information but cost of training time increase with increase in dimensions. The accuracy increases in general with dimensions (for a specific task and for a specific corpus size) and then tapers off with diminishing returns.  FLAG NEEDED
 - -N: Number of nodes in each ego-network/sub-graphs of neighbouring words for creating the clusters/senses of a word. For example, if we set this value to be 200, while creating senses for a word, 200 nearest neighbours words will be considered for creating senses (default is 200). Advantage of using small valus of N are that the training time is reduced and we get less noisy senses, however, due to fewer neighbours considered, few senses can be missed out. Advantage using large values of N is that we can get more variety of senses (it is not deterministic), however, training time will increase and there is a chance to get noise sense as we are considering large number of neighbours to create senses. In sensegram official paper and implementation, they have used N=200. FLAG NEEDED
 - -n: Maximum number of edges a node can have in the network (default is 200). It does not effect the results, but use N and n equal to keep consistent filenames. FLAG NEEDED. 
 - -num_of_models: Number of models we need to train for a single corpus. Number of models to be trained (default 1). FLAG NEEDED.
 - -output_models_base_path: Base path of the directory where we need to store the models. (default model/). This is the where all the trained models will be stored. Recommended to use the default path. FLAG NEEDED.

## Outputs:
Trains the models and store them under _output_models_base_path/corpus_name_

# Generate senses from a list of word:
## Command:
```
python 1_syed_load_word_and_sense_vectors_for_senses.py 
```
## Inputs:
Set the following variables in the file before executing the command.
 - corpus_name 
 - model_base_dir [Directory in which the models are present]

Effective directory path used to load the models is: 
_model_base_dir/corpus_name_

 - The code asks for the path of file which contains words to get senses.

### Example 
word
<br>because
<br>belong

## Outputs:
It creates a directory _word_senses/corpus_name_ which contains csv files for each word provided in the input.

### Example
![image](https://user-images.githubusercontent.com/22868291/151033691-18e7dc3e-9547-4b63-94db-5d0a08f0e87d.png)


# Reduce the dimensions of word vectors for better visualization using PCA.
## Command:
```
python3 1_reduce_dimensions_wordvectors.py 
```
## Inputs:
 - corpus_name [Name of the corpus. Models of which will be used to generate word vectors. Models should be stored in the directory model/]
 - words_to_get_senses_filename [Path to the file which contains the list of words. Each line should have only one word.]
 - path_to_save_file [Line 144-147. Path where to save the PCA output file.]

### Example 
word
<br>because
<br>belong

## Outputs:
 - Writes a CSV file on the specified path after performing PCA. It has columns: word, X,	Y,	Z,	word_vector
### Example
word	X	Y	Z	word_vector
know#1	0.042678845870709	1.84172737636189	0.219532032706539	[-0.00445312  0.02521706  0.06794521 ... -0.0051711  -0.05367299   0.0570246 ]
it#1	-1.00615402843201	0.068563908244432	0.810797539348226	[ 0.01901305  0.01266508  0.06339029 ... -0.0460179   0.00458821   0.03217988]
because#1	-0.219415038384593	0.919801487363445	0.596360836775124	[ 0.02768713  0.00505522  0.07169732 ...  0.01527843 -0.00958562   0.02164705]
pain#1	0.11036348385649	-0.72624835987096	0.326537689271145	[-0.10797457  0.03026501 -0.05312319 ... -0.03365547 -0.00602931  -0.00088333]
remember#1	0.159423898008495	1.76650607600389	0.10264954091651	[-0.03116777  0.00351258  0.04525061 ... -0.01563067 -0.04849707   0.02050507]
ego#1	1.59470951533283	-0.914308073209488	0.311534240733845	[-0.08355454 -0.0006044  -0.0421834  ... -0.00962467  0.01350354  -0.01859214]
![image](https://user-images.githubusercontent.com/22868291/151032689-80cfc0a4-ef16-46cd-944d-1bf847c2ef5a.png)


# Bag of words:
## Command:
```
python 1_bag_of_words.py
```
## Inputs:
 - corpus_name [Path to the folder which contains the models we need to use e.g. corpus_name = "/media/sensegram/38d2342b-a798-4821-a3b9-16efbcf34f12/model/sample.txt"]
 - word_to_process_for_BoW_FILENAME [Path to the file to be processed e.g. word_to_process_for_BoW_FILENAME = "words_to_get_senses/holliday.txt"]

### Example:
(words_to_get_senses/works.txt)
<br>cities: paris#1, london#1, lahore#1, dubai#1
<br>furniture: table#1, chair#1, seat#1, bed#1
<br>pets: dog#1, cat#1, fish#1, rabbit#1,
<br>food: bread#1, burger#1, pizza#1, hotdog#1
<br>fruits: apple#1,orange#1, banana#1
<br>winter_weather: snow#1, hail#1, ice#1, cold#1
<br>colours: red#1, blue#1, black#1, green#1, white#1, pink#1

## Outputs:
This file writes three files.
 1. PCA file as above in reduce_dimensions_wordvectors with extra column BoW. It has columns: BoW, word, X,	Y,	Z,	word_vector
 2. Aggregated PCA file for each bag of word. It has columns: BoW,	X,	Y,	Z,	similar_words
 3. Distance matrix between all the bag of words. 

### Example
#### 1.
BoW	word	X	Y	Z	word_vector
<br>furniture	desk#1	1.20031621055252	0.325482087659498	-0.398094754779704	[ 0.09106176  0.0579374   0.02598179 ...  0.04057969  0.03202484  -0.01632051]
<br>furniture	bus#1	1.20007876152443	0.098693501039042	-0.255482184779384	[ 0.07297327  0.05974918  0.04270832 ...  0.01397101 -0.00664327  -0.00812657]
<br>furniture	ride#1	1.15872225838677	0.460562932932283	-0.096189247875183	[ 0.10819038  0.07170317  0.02553471 ...  0.01673624  0.00511917  -0.0291835 ]
<br>furniture	kitchen#1	1.03118766614974	0.369546799492503	-0.472118439933155	[ 0.11344537  0.06694352  0.00466962 ...  0.03174785  0.02868408  -0.00885597]
<br>furniture	dark#1	0.546288542595897	0.449046659920669	0.27187627117154	[ 0.03741607  0.03636116 -0.03560129 ... -0.0062227  -0.01622491   0.02159884]
<br>furniture	alarm#1	0.410922275426199	-0.013808111110877	-0.375357141603995	[ 0.06155611  0.03996786  0.01621729 ...  0.03550377  0.00671058  -0.01469874]

#### 2.
BoW	X	Y	Z	similar_words <br><br>
food	-0.661367628807547	0.619068444717633	-0.021656926517855	{'gin#1', 'chicken#1', 'beef#1', 'banana#1', 'garlic#1', 'toast#1', 'roast#1', 'cheese#1', 'burger#1', 'coke#1', 'steak#1', 'sandwich#1', 'fruit#1', 'veg#1', 'eggs#1', 'pie#1', 'sandwiches#1', 'sausage#1', 'bread#1', 'crisps#1', 'sauce#1', 'fish#1', 'tonic#1', 'diet#1', 'baked#1', 'pasta#1', 'biscuits#1', 'soup#1', 'chilli#1', 'cakes#1', 'choc#1', 'beans#1', 'bacon#1', 'jam#1', 'chips#1', 'curry#1', 'fresh#1', 'cream#1', 'butter#1', 'egg#1', 'champagne#1', 'vodka#1', 'salad#1', 'yum#1', 'potato#1', 'ice#1', 'cake#1', 'pizza#1', 'milk#1', 'sugar#1', 'meat#1'}
<br><br>fruits	-0.529618390762956	-0.111580586794987	-0.052685197260453	{'raspberry#1', 'whisky#1', 'cookie#1', 'cider#1', 'banana#1', 'raw#2', 'bean#1', 'drizzle#1', 'buttons#1', 'cookies#1', 'tonic#1', 'protein#1', 'peanut#1', 'vinegar#1', 'lime#1', 'burger#1', 'olive#1', 'onion#1', 'tin#1', 'plastic#1', 'earl#1', 'belly#1', 'thai#1', 'flavoured#1', 'spray#1', 'yoghurt#1', 'raw#1', 'tart#1', 'custard#1', 'tomato#1', 'nut#1', 'linen#1', 'tomatoes#1', 'gloves#1', 'pots#1', 'salmon#1', 'crisps#1', 'cherry#1', 'polish#1', 'inch#1', 'cereal#1', 'plate#1', 'soya#1', 'strawberry#1', 'apple#1', 'perfume#1', 'chinese#1', 'cucumber#1', 'icecream#2', 'juice#1', 'knife#1', 'chip#1', 'loaf#1', 'coconut#1', 'smoothie#1', 'baked#1', 'leather#1', 'belt#1', 'mash#2', 'orange#1', 'pear#1', 'pepper#1', 'espresso#2', 'tan#2', 'freezer#1', 'choc#1', 'espresso#1', 'salt#1', 'booze#1', 'caramel#1', 'slices#1', 'yellow#1', 'dish#1', 'spicy#1', 'dairy#1', 'iced#1', 'oil#2', 'corn#1', 'lavender#1', 'rice#1', 'frozen#1', 'lolly#1', 'homemade#1', 'biscuit#1'}
<br><br>furniture	0.866997900385184	0.211784072487465	-0.25526569665062	{'desk#1', 'screen#2', 'bus#1', 'ride#1', 'kitchen#1', 'dark#1', 'alarm#1', 'walk#1', 'drive#2', 'floor#1', 'mile#1', 'hole#1', 'outside#1', 'left#2', 'seat#1', 'bathroom#1', 'empty#2', 'bike#1', 'fire#1', 'screen#1', 'duvet#1', 'bed#1', 'pool#1', 'winter#1', 'corner#1', 'shop#1', 'chair#1', 'oven#1', 'flat#1', 'river#1', 'machine#1', 'window#1', 'room#1', 'sitting#1', 'light#1', 'step#1', 'way#2', 'sofa#1', 'fridge#1', 'washing#1', 'door#1', 'house#1', 'roof#1', 'bar#1', 'walking#2', 'tree#1', 'place#2', 'wall#1', 'pub#1', 'box#1', 'air#1', 'bedroom#1', 'journey#1', 'garden#1', 'table#1', 'pack#1', 'pjs#1', 'lights#1', 'doors#1', 'office#1', 'storm#1', 'car#1', 'heat#1'}

#### 3.
![image](https://user-images.githubusercontent.com/22868291/151029874-a2d6e0a6-24a8-4397-a831-e12a23072feb.png)


# Steps for installation
(https://docs.anaconda.com/anaconda/install/linux/#installation)
## Install the anaconda
 - Download the anaconda from link: https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
 - Enter the following on terminal to install Anaconda ```bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh```
 - The installer prompts "In order to continue the installation process, please review the license agreement." Click Enter to view license terms.
 - Scroll to the bottom of the license terms and enter “Yes” to agree.
 - Press Enter
 - The installer prompts “Do you wish the installer to initialize Anaconda3 by running conda init?” Choose “yes”.
 - The installer finishes and displays “Thank you for installing Anaconda<2 or 3>!”
 - Close and open your terminal window for the installation to take effect.
 
## Placing the environment
After the successful installation of anaconda:
 - Environment zip file is place on path: /home/sensegram/sense_bespoke/sensegram_env.zip
 - Place the provided environment zip file in: ~/anaconda/envs/
   - Absoulte path on sensegram machine is: /home/sensegram/anaconda/envs
 - Unzip the file here. 

## Using the environment
After successfully following above steps, the environment is ready to use. It can be activated using the command:
```
conda activate sensegram_env
```
