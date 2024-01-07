# Automatic Short Answer Grading

This code is a Automatic Short Answer Grading (ASAG) system on the Mohler (Texas) dataset.

## Mohler Dataset

`data/` contains two files in csv format:

Data.csv consists of a total of 2442 student answers as a response to around 87 computer science questions collected over a period of 10 assignments and 2 tests.
The scoring is continuous over a range of 0-5 and every answer is graded by two human evaluators. We consider their average as the gold standard score.

QA1.csv consists of just the questions and the model answers, thus containing lesser entries.

## Download embeddings
Download the Baroni embeddings [here](https://osf.io/489he/wiki/dcp_cbow/) and extract them into the `./data/` directory. 

Afterwards the `./data/` directory should look like:

```
data/
    Data.csv
    QA1.csv
    EN-wform.w.5.cbow.neg10.400.subsmpl.txt
```

## Install Packages

Packages used in this project can be installed with the following command:
`pip install -r requirements.txt`

## Experimentation

### Preprocessing
1. Text normalization includes converting text to lowercase, removing non-alphanumeric characters, tokenizing, and filtering out stopwords.
2. Creating a correct word pool by parsing through a collection of texts, excluding words shorter than two characters.
3. Identification of potentially incorrect words by comparing against WordNet and using a spellchecker.
4. Mapping potentially incorrect words to their potential correct counterparts:
5. Utilizing Levenshtein and fuzzy matching ratios for potential matches.
6. Refine the word mapping pool by spellchecking, word splitting, and checking against word pools.
7. Replacement of identified incorrect words in the DataFrame column with their potential correct versions based on the generated dictionary.

### Feature Extraction
1. Generate Sum Of the Word Embeddings (SOWE) for all the answers in the dataset when calculating the cosine similarity.
2. These embeddings are created using the Baroni embeddings.
3. We generate features for each student answer by calculating cosine similarities between the reference answer and every student answer. The features are cosine similarity, alignment score, length ratio, eucledian distance, fuzzy features. You can find the detailed code for feature extraction in `feature_extraction.py`.
5. We use these features for training the regression models.

### Training and Testing
1. We split the Mohler data into 75%-25% training and testing data.
2. We use the training data to train on regression models namely, Random Forest Regressor, Ridge regression and a Neural Network.
3. We use these trained models, to predict the grades of test data and generate the results.

## Training model

To train the model, you first need to download the embeddings of Baroni so that they can be used for feature extraction.

## Run Training

To run training, run:

`bash main.sh`

## References
<a id="1">[1]</a> 
[Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors](https://aclanthology.org/P14-1023) (Baroni et al., ACL 2014)