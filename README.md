# Automatic Short Answer Grading

This code is a Automatic Short Answer Grading (ASAG) system on the Mohler (Texas) dataset.

## Mohler Dataset

`data/` contains two files in csv format:

`data/
    # (2442 entries)
    Data.csv`

    # (87 entries)
    QA1.csv`

Data.csv consists of a total of 2442 student answers as a response to around 87 computer science questions collected over a period of 10 assignments and 2 tests.
The scoring is continuous over a range of 0-5 and every answer is graded by two human evaluators. The scores are graded from 0(not correct) to 5(totally correct). The grades are allocated by two evaluators. 
We consider their average as the gold standard score.

QA1.csv consists of just the questions and the model answers, thus containing lesser entries.

## Install Packages

Packages used in this project can be installed with the following command:
`pip install -r requirements.txt`

## Experimentation

Preprocessing
1. Tokenization is applied on both student answer and reference answers
2. Lemmatization and stopword removal are neglected consciously, to assess the performance of the transfer learning models
3. spell checker is also neglected, assuming that the graders had deducted the scores for misspelled words

Feature Extraction
1. Generate Sum Of the Word Embeddings (SOWE) for all the answers in the dataset when calculating the cosine similarity.
2. These embeddings are created using the Baroni embeddings.
3. We generate features for each student answer by calculating cosine similarities between the reference answer and every student answer. The features are cosine similarity, alignment score, length ratio, eucledian distance, fuzzy features. You can find the detailed code for feature extraction in `feature_extraction.py`.
5. We use these features for training the regression models.

Training and Testing
1. We split the Mohler data into 75%-25% training and testing data.
2. We use the training data to train on regression models namely, Random Forest Regressor, Ridge regression and a Neural Network.
3. We use these trained models, to predict the grades of test data and generate the results.

## Training model

To train the model, you first need to download the embeddings of Baroni so that they can be used for feature extraction.

## Download Baroni embeddings
Please download the Baroni embeddings [here](https://osf.io/489he/wiki/dcp_cbow/) and put them in `/data`.

## Run Training

To run training, run:

`bash scripts/main.sh`
