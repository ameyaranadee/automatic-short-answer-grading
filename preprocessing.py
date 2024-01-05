import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
wpt = nltk.WordPunctTokenizer()
stop_words = stopwords.words('english')

from autocorrect import Speller
import wordninja
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from spellchecker import SpellChecker
from Levenshtein import ratio


# normalizing document using regex and stop words
punctuations = ['(',')','-lrb-','.',',','-','?','!',';','_',':','{','}','[','/',']','...','"','\'', '>', '<','-rrb-', '))', '!;', '.,', '', '!,', '!<', '!=', "'+'", '++', '++,', '++.', ',<', '--', '..', '....', '...<', '.<', '//', '0x000000', '2n', '3n', '::=', ':<', ';;', ';<', ';=', '==']

def normalize_document(df, text_colname, que_colname=None, qd=False):
    for i in range(len(df)):
        doc = str(df[text_colname][i])
        # replace text that is not string with space, re.I - ignore case, re.A - perform ASCII matching
        # print(doc)
        doc = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", doc)
        # print(doc)
        doc = doc.lower()
        doc = doc.strip()
        tokens = wpt.tokenize(doc)
        # print(tokens)
        # take token if it is not in stop words
        if qd:
            q_tokens = wpt.tokenize(df[que_colname][i])
            filter_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token != 'br' and token not in punctuations and token not in q_tokens]
        else:
            filter_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token != 'br' and token not in punctuations]

        df[text_colname][i] = ' '.join(filter_tokens)
    # return df

# to create correct word pool
def get_correct_words(correct_word_pool = set(), texts = [[]]):
    for text in texts:
        text1 = text.split(' ')
        for t in text1:
            correct_word_pool.add(t)

    correct_pool = correct_word_pool.copy()

    for c in correct_pool:
        if len(c) <= 2: 
            correct_word_pool.remove(c)

# to create a wrong word pool - custom function to check wordnet and spellchecker
def spell_checker(wrong_word_pool = set(), texts = [[]], correct_word_pool = set()):
    spell = SpellChecker()
    for text in texts:
        text1 = text.split(' ')
        temp_list= []
        for t in text1:
            if (not wordnet.synsets(t)) and (t not in correct_word_pool):
                temp_list.append(t)

        text0 = spell.unknown(temp_list)
        for i in text0:
            if i not in correct_word_pool:
                wrong_word_pool.add(i)

    if '' in wrong_word_pool:
        wrong_word_pool.remove('')

def check_wrong_pool(wrong, pool):
    wrong_word_pool_copy = wrong.copy()
    for w in wrong_word_pool_copy:
        if w in pool:
            wrong.remove(w)  

#creating wrong to correct dictionary 
def wrong_to_correct(df, wrong_word_pool = set(), texts = [[]], correct_word_pool = set(), score_colname = ""):
    freq_full_ans = {}
    pool_dictionary = {}

    # We check answers that have full grades and take words that exist in those answers. if frequency of this > 4 then remove from wrong word pool
    for i in range(len(df)):
        if df[score_colname][i] == 5:
            t = texts[i]
            t1 = t.split(' ')
            for t2 in t1:
                if t2 in freq_full_ans: 
                    freq_full_ans[t2] += 1
                else: 
                    freq_full_ans[t2] = 1

    for freq in freq_full_ans:
        if freq_full_ans[freq] >= 4 and len(freq) >= 4 and freq in wrong_word_pool: 
            correct_word_pool.add(freq)

    # using levenshetin ratio and fuzzy wuzzy ratio to create dictionary (wrong: correct)
    for w in wrong_word_pool:
        for c in correct_word_pool:
            levenshetin_ratio = ratio(w, c)
            fuzzy_ratio = fuzz.ratio(w, c)

            if (w[0] == c[0]) and (levenshetin_ratio >= 0.80 or fuzzy_ratio >= 80):
                pool_dictionary[w] = c
    
    check_wrong_pool(wrong=wrong_word_pool, pool=pool_dictionary)

    spell1 = SpellChecker()
    spell2 = Speller(lang='en') #autocorrect library
    for w in wrong_word_pool:
        a = spell1.correction(w)
        b = spell2(w)
        if a == b and wordnet.synsets(a):
            pool_dictionary[w] = a

    check_wrong_pool(wrong=wrong_word_pool, pool=pool_dictionary)

    for w in wrong_word_pool:
        t = wordninja.split(w)
        c = 0
        for t1 in t:
            # print(t1)
            if (wordnet.synsets(t1) or t1 in correct_word_pool or t1 in stop_words or t1 == "ptr") and len(t1) >= 2:
                c+=1
        if c >= len(t)*0.80:
            temp = " ".join(t)
            pool_dictionary[w] = temp

    return pool_dictionary

# correcting words in the dataframe column (texts)
def replace_wrong_words(pool_dictionary, df, col_name):
    for i in range(len(df)):
        text = df[col_name][i]
        text1 = text.split(' ')

        for j in range(len(text1)):
            if text1[j] in pool_dictionary:
                text1[j] = pool_dictionary[text1[j]]

        df[col_name][i] = " ".join(text1)
    

    

