from fuzzywuzzy import fuzz
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize
wpt = nltk.WordPunctTokenizer()

def cosine(u, v):
    return (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

#For baroni et al embeddings:- Word2Vec

def computeVecSum(vectors):
  n = len(vectors)
  d = 399

  s = []
  for i in range(d):
    s.append(0)
  s = np.array(s)

  for vec in vectors:
    s = s + np.array(vec)

  return (s)

def cos_sim(df, text_colname, ans_colname, emb):
  cos = []

  for t in range(len(df)):
    model_answer = word_tokenize(list(df[ans_colname])[t])
    if type(df[text_colname][t]) == str:
      texts = wpt.tokenize(list(df[text_colname])[t])
      word_vecM = [emb[i] for i in model_answer if i in emb]
      word_vecR = [emb[i] for i in texts if i in emb]
      sent_vecM = computeVecSum(word_vecM)
      sent_vecR = computeVecSum(word_vecR)
      cos.append(cosine(sent_vecM,sent_vecR))
    else:
      cos.append(0)

  return cos

def cos_wm(df, qn_colname, text_colname, ans_colname, score_colname, emb):
  cos = []

  for i in range(len(df)):
    sim = []
    sim_c = 0
    model_answer = word_tokenize(list(df[ans_colname])[i])
    word_vecM = [emb[k] for k in model_answer if k in emb]
    sent_vecM = computeVecSum(word_vecM)
    for j in range(len(df)):
      if i == j:
        continue
      elif df[qn_colname][i] == df[qn_colname][j] and df[score_colname][j] == 5 and type(df[text_colname][j]) == str:
        texts = wpt.tokenize(list(df[text_colname])[j])
        word_vecR = [emb[k] for k in texts if k in emb]
        sent_vecR = computeVecSum(word_vecR)
        sim.append(cosine(sent_vecM,sent_vecR))
        sim_c+=5
    if type(df[text_colname][i]) == str:
        texts = wpt.tokenize(list(df[text_colname])[i])
        word_vecR = [emb[k] for k in texts if k in emb]
        sent_vecR = computeVecSum(word_vecR)
        c = (sim_c*cosine(sent_vecM,sent_vecR) + sum(sim))/(2*sim_c)
    elif sim_c != 0:
      c = sum(sim)/(2*sim_c)
    else:
      c = 0
    cos.append(c)

  return cos

def alignment(df, text_colname, ans_colname, emb):
  align = []

  for i in range(len(df)):
    if type(df[text_colname][i]) == str:
      test_model_answer = word_tokenize(list(df[ans_colname])[i])
      test_text = word_tokenize(list(df[text_colname])[i])

      x, y = [], []

      for ma in test_model_answer:
        for tt in test_text:
          if ma in emb and tt in emb:
            ma_embedding = emb[ma]
            tt_embedding = emb[tt]

            cos_similarity = cosine(ma_embedding, tt_embedding)

            if cos_similarity >= 0.4:
              x.append(ma)
              y.append(tt)

      alignment_score = (len(set(x)) + len(set(y))) / (len(set(test_model_answer)) + len(set(test_text)))
      align.append(alignment_score)
    else:
      align.append(0)
  return align

def length(df, text_colname, ans_colname):
  length_ratio = []

  for i in range(len(df)):
    if type(df[text_colname][i]) == str:
      test_model_answer = word_tokenize(list(df[ans_colname])[i])
      test_text = word_tokenize(list(df[text_colname])[i])

      lr = len(set(test_text)) / len(set(test_model_answer))
      length_ratio.append(round(lr, 3))

    else:
      length_ratio.append(0)

  return length_ratio

def eucledian(df, text_colname, ans_colname, emb):
  eucledian_distance = []

  for t in range(len(df)):
    model_answer = word_tokenize(list(df[ans_colname])[t])
    if type(df[text_colname][t]) == str:
      texts = wpt.tokenize(list(df[text_colname])[t])
      word_vecM = [emb[i] for i in model_answer if i in emb]
      word_vecR = [emb[i] for i in texts if i in emb]
      sent_vecM = computeVecSum(word_vecM)
      sent_vecR = computeVecSum(word_vecR)
      ed = round(np.linalg.norm(sent_vecM-sent_vecR), 3)

    else:
      ed=0

    eucledian_distance.append(ed)

  return eucledian_distance

def fuzzy_features(df, text_colname, ans_colname):
  fuzzy_ratio = []
  fuzzy_partial_ratio = []
  fuzzy_token_sort = []
  fuzzy_token_set = []

  for t in range(len(df)):
    model_answer = df[ans_colname][t]
    if type(df[text_colname][t]) == str:
      texts = df[text_colname][t]

      fuzzy_r = fuzz.ratio(model_answer, texts)
      fuzzy_pr = fuzz.partial_ratio(model_answer, texts)
      fuzzy_tsort = fuzz.token_sort_ratio(model_answer, texts)
      fuzzy_tset = fuzz.token_set_ratio(model_answer, texts)

    else:
      fuzzy_r = 0
      fuzzy_pr = 0
      fuzzy_tsort = 0
      fuzzy_tset = 0

    fuzzy_ratio.append(fuzzy_r / 100)
    fuzzy_partial_ratio.append(fuzzy_pr / 100)
    fuzzy_token_sort.append(fuzzy_tsort / 100)
    fuzzy_token_set.append(fuzzy_tset / 100)

  return [fuzzy_ratio, fuzzy_partial_ratio, fuzzy_token_sort, fuzzy_token_set]