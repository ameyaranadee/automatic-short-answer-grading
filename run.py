import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from feature_extraction import cos_sim, cos_wm, alignment, length, eucledian, fuzzy_features
from preprocessing import normalize_document, get_correct_words, spell_checker, wrong_to_correct, replace_wrong_words
from args import get_parser
from embeddings import readBaroni

from pathlib import Path
from packaging import version

proj_dir = Path(__file__).resolve().parent.parent

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.verbose = True
        
        if not os.path.isdir(self.args.download_dir):
            if self.verbose:
                print("\nCreating download directory")
            os.makedirs(self.args.download_dir)

    def preprocess(self):
        dataf = pd.read_csv(self.args.dataframe)

        if self.args.question_demoting:
            if self.verbose:
                print("\nNormalizing texts column by demoting questions")
            normalize_document(dataf, self.args.text_colname, self.args.que_colname, qd=True)
        else:
            if self.verbose:
                print("\nNormalizing texts column")
            normalize_document(dataf, self.args.text_colname)

        normalize_document(dataf, self.args.ans_colname)

        correct_set = set()
        if self.args.qa_dataframe:
            print('Taking QA1 dataframe')
            qa_df = pd.read_csv(self.args.qa_dataframe)
            normalize_document(qa_df, self.args.que_colname)
            normalize_document(qa_df, self.args.ans_colname)
            get_correct_words(correct_set, qa_df[self.args.que_colname].unique())
            get_correct_words(correct_set, qa_df[self.args.ans_colname].unique())
        else:
            get_correct_words(correct_set, dataf[self.args.que_colname].unique())
            get_correct_words(correct_set, dataf[self.args.ans_colname].unique())

        if self.verbose:
            print('Length of correct word pool: ', len(correct_set))

        wrong_set = set()
        spell_checker(wrong_set, dataf[self.args.text_colname], correct_set)
        if self.verbose:
            print('Length of wrong word pool: ', len(wrong_set))

        if self.verbose:
            print("Creating a word pool dictionary")
        words_pool = wrong_to_correct(dataf, wrong_set, dataf[self.args.text_colname], correct_set, self.args.score_colname)
        if self.verbose:
            print('Length of word pool dictionary', len(words_pool))

        if self.verbose:
            print("Replacing wrong words in the text column")
        replace_wrong_words(words_pool, dataf, self.args.text_colname)

        if self.verbose:
            print("Saving the processed data csv to download directory")
        dataf.to_csv(self.args.download_dir+"processed_df.csv", index=False)
        
        return dataf


    def embed(self):
        if self.args.w2v_baroni:
            if self.verbose:
                print("\nLoading Baroni's Word2Vec embeddings")
            w2v = readBaroni(self.args.w2v_baroni_dir)

        return w2v
    
    def feature_ex(self, emb):

        data = self.preprocess()
        
        if self.verbose:
            print("\nExtracting features")

        cos_normal = cos_sim(data, self.args.text_colname, self.args.ans_colname, emb)
        cos_wm_sim = cos_wm(data, self.args.qnum_colname, self.args.text_colname, self.args.ans_colname, self.args.score_colname, emb)
        alignment_score = alignment(data, self.args.text_colname, self.args.ans_colname, emb)
        length_ratio = length(data, self.args.text_colname, self.args.ans_colname)
        eucledian_distances = eucledian(data, self.args.text_colname, self.args.ans_colname, emb)
        fuzz_feat =  fuzzy_features(data, self.args.text_colname, self.args.ans_colname)
        fuzzy_ratio = fuzz_feat[0]
        fuzzy_partial_ratio = fuzz_feat[1]
        fuzzy_token_sort = fuzz_feat[2]
        fuzzy_token_set = fuzz_feat[3]

        scores = data[self.args.score_colname].tolist()

        features_df = pd.DataFrame({'cos_sim': cos_normal, 'cos_wm_sim' : cos_wm_sim, 'alignment': alignment_score, 'length_ratio' : length_ratio, 'distances': eucledian_distances, 'fuzzy_ratio': fuzzy_ratio, 'fuzzy_partial_ratio': fuzzy_partial_ratio, 'fuzzy_token_sort': fuzzy_token_sort, 'fuzzy_token_set': fuzzy_token_set, 'Score': scores})
        
        if self.verbose:
            plt.figure(figsize=(12, 12))
            heatmap = sns.heatmap(features_df.corr(), vmin=-1, vmax=1, annot=True)
            heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

        if self.verbose:
            print("Saving the feature data csv to download directory")
        features_df.to_csv(self.args.download_dir+"feature_df.csv", index=False)

        return features_df
    
    def train(self):
            
        emb = self.embed()
        fd = self.feature_ex(emb)
        fd.fillna(0,inplace=True)

        # setting arguements for question demoting
        if self.args.qd_train:
            args.question_demoting = True
            fd1 = self.feature_ex(emb)
            fd1.fillna(0,inplace=True)
            fd = pd.concat([fd, fd1], axis=1)
            fd.columns = ['cos_sim', 'cos_wm_sim', 'alignment', 'length_ratio', 'distances', 'fuzzy_ratio', 'fuzzy_partial_ratio', 'fuzzy_token_sort', 'fuzzy_token_set', 'Score', 'cos_sim_qd', 'cos_wm_sim_qd', 'alignment_qd', 'length_ratio_qd', 'distances_qd', 'fuzzy_ratio_qd', 'fuzzy_partial_ratio_qd', 'fuzzy_token_sort_qd', 'fuzzy_token_set_qd', 'Score_qd']
            fd.drop('Score_qd', axis=1, inplace=True)
        
        fd.to_csv(self.args.download_dir+"final_features.csv", index=False)
        
        X = fd.drop('Score', axis=1)
        y = fd.Score

        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=self.args.train_test_split, random_state = 12)

        if self.verbose:
            print("\nTraining the model")

        if self.args.regressor == "RandomForestRegressor":
            model = RandomForestRegressor(max_depth=self.args.max_depth, random_state=13)
            
        elif self.args.regressor == "Ridge":
            model = Ridge(alpha=self.args.alpha)
        
        elif self.args.regressor == "MLPRegressor":
            model = MLPRegressor(random_state=17, max_iter=self.args.max_iter)
            
        model = model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        score = model.score(xtest,ytest)
        mse = mean_squared_error(ytest,ypred)
        corr, _ = pearsonr(ytest, ypred)

        print('\nTraining Accuracy:')
        print('Pearsons correlation for '+self.args.regressor+': %.3f' % corr)
        print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.3f}".format(score, mse, np.sqrt(mse)))

def main_worker(gpu, args):
    
    trainer = Trainer(args)
    trainer.train()
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main_worker(0, args)