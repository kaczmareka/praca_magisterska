#sentiment - graph-based
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import numpy as np
# import pandas as pd
# import time
# from sklearn.metrics import balanced_accuracy_score

# sentiment functions

# graph-based
def find_ents(doc):
  list_of_ents=[]
  if doc.ents:
    for ent in doc.ents:
      list_of_ents.append(ent.text)
  return list_of_ents

def get_words_for_nodes(text, nlp, list_ners=[], lemmatization=False, max_nodes=0):
  doc1 = nlp(text)
  # find ners
  if type(list_ners)==list and len(list_ners)>0:
    ners=list_ners
  else:
    ners = find_ents(doc1)
  if lemmatization:
    lemmatized_tokens=[token.lemma_ for token in doc1]
    text = ' '.join(lemmatized_tokens)
    doc1=nlp(text)
  tags=['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
  nouns_verbs_etc = [token.text for token in doc1 if token.tag_ in tags]
  text2=" ".join([str(word).replace(" ", "_") for word in list(doc1)])
  all_nodes=[str(word).replace(" ", "_") for word in list(set(ners+nouns_verbs_etc))]
  if max_nodes !=0:
    word_dict = {word: 0 for word in all_nodes}
    for word in text2.split():
      if word in word_dict.keys():
        word_dict[word] +=1
    word_dict_sorted=sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
    all_nodes = [item[0] for item in word_dict_sorted]
  else:
    word_dict=[]
  return all_nodes,text2, ners, word_dict

def get_weights_of_edges(text, words, max_distance=20, ner_list=[]):
  # if ner_list not empty, we should calculate the distances only between ners and other words
  # and do not between other words and other words

  list_from_text = text.split()
  weight_matrix=np.zeros((len(words), len(words)))
  occurences_list=np.zeros(len(words))
  for i, word1 in enumerate(list_from_text):
    j=i+1
    try:
      index1=words.index(word1)
      occurences_list[index1]+=1
    except ValueError:
      pass
    while j<=i+max_distance and j<len(list_from_text):
      word2=list_from_text[j]
      if len(ner_list) != 0 and (word1 in ner_list or word2 in ner_list):
        try:
          index1=words.index(word1)
          index2=words.index(word2)
          distance=j-i
          if distance!=0:
            inv_distance=1/distance
            weight_matrix[index1][index2]+=inv_distance
        except ValueError:
          pass
      elif len(ner_list)==0:
        try:
          index1=words.index(word1)
          index2=words.index(word2)
          distance=j-i
          if distance!=0:
            inv_distance=1/distance
            weight_matrix[index1][index2]+=inv_distance
        except ValueError:
          pass
      else:
        pass
      j+=1
  return weight_matrix, occurences_list

def calculate_sentiment_for_nodes(text, compound=False):
  analyzer = SentimentIntensityAnalyzer()
  # Loop through the words/ bigrams from 1. and get the sentiment scores for each one
  sentiment_scores=np.zeros(len(text))
  for i, text in enumerate(text):
    scores=analyzer.polarity_scores(text)
    if compound:
      sentiment_scores[i]=scores['compound']
    else:
      if scores['neg']==1.0:
        sentiment_scores[i]=-1
      elif scores['pos']==1.0:
        sentiment_scores[i]=1
  return sentiment_scores

def weighted_sentiment_func(weight_matrix, occ_list, sentiment_scores, words, ner_list=[]):
  sum_columns_weights=np.sum(weight_matrix, axis=0)
  sum_rows_weights=np.sum(weight_matrix, axis=1)
  if len(ner_list)!=0: #if ners are predefined, calculate only for them
    for i, word in enumerate(words):
      if word not in ner_list:
        sum_columns_weights[i]=0
        sum_rows_weights[i]=0
  sum_all_weights=sum_columns_weights+sum_rows_weights
  count_nonzero_weights_columns=np.count_nonzero(weight_matrix, axis=0)
  count_nonzero_weights_rows=np.count_nonzero(weight_matrix, axis=1)
  count_nonzero_weights=count_nonzero_weights_columns+count_nonzero_weights_rows
  count_nonzero_weights[count_nonzero_weights==0]=1
  mean_columns_weights=sum_all_weights/count_nonzero_weights
  total_weight=mean_columns_weights*occ_list
  return total_weight*sentiment_scores

def calculate_sentiment_of_text(weighted_sentiment, threshold=0.05, output_number=False):
  sum_sentiment=np.sum(weighted_sentiment)
  if output_number:
    if sum_sentiment>threshold:
      return 1
    elif sum_sentiment<-threshold:
      return -1
    else:
      return 0
  else:
    if sum_sentiment>threshold:
      return "positive"
    elif sum_sentiment<-threshold:
      return "negative"
    else:
      return "neutral"

def graph_sentiment_analysis(text, nlp, lemmatization=False, max_distance=20, ner_list=[], compound=False, output_number=False, calculate_overall_score=1, threshold=0.05, max_nodes=0):
  words_all, text_all, ners_all, words_dict = get_words_for_nodes(text, lemmatization=lemmatization, nlp=nlp, max_nodes=max_nodes)
  weight_matrix_all, occ_list_all=get_weights_of_edges(text_all, words_all, max_distance=max_distance, ner_list=ner_list)
  sentiment_scores_all=calculate_sentiment_for_nodes(words_all, compound=compound)
  weighted_sentiment_all=weighted_sentiment_func(weight_matrix_all, occ_list_all, sentiment_scores_all, words=words_all, ner_list=ner_list)
  if calculate_overall_score==1:
    return calculate_sentiment_of_text(weighted_sentiment_all, output_number=output_number, threshold=threshold)
  elif calculate_overall_score==0:
    return dict(zip(words_all, weighted_sentiment_all))
  else:
    words_all.append("overall_sentiment")
    weighted_sentiment_all=list(weighted_sentiment_all)
    weighted_sentiment_all.append(calculate_sentiment_of_text(weighted_sentiment_all, output_number=True))
    return dict(zip(words_all, weighted_sentiment_all))

def run_actaware_preprocessed(articles):
  nlp = spacy.load('en_core_web_sm')
  nlp.add_pipe("merge_noun_chunks")
  # actaware_df=pd.DataFrame({'Text': list_of_articles, 'Sentiment': ""})
  answers=["" for _ in range(len(articles))]
  for i in range(len(articles)):
    text_to_check=articles[i]
    answers[i]=graph_sentiment_analysis(text_to_check, calculate_overall_score=1, nlp=nlp, threshold=0.0, compound=True)
  # return balanced_accuracy_score(df_gt['Sentiment'], actaware_df['Sentiment'])
  return answers
