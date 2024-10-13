from sklearn.metrics import balanced_accuracy_score
from sentence_transformers import SentenceTransformer, SimilarityFunction
import pandas as pd
from src.sentiment_roberta import run_roberta_sentiment
from src.sentiment_graph_based import run_actaware_preprocessed

from src.config import(
  PATH_GT_DATA,
  PATH_NOT_PREPROCESSED,
  PATH_4O,
  PATH_REGEX,
  PATH_HUMAN
)

def evaluate_category(data, category_predicted):
  """
  Compute balanced accuracy on predicted categories.

  Args:
    data (pd.DataFrame): ground truth
    category_predicted (list): predicted categories for articles
  """
  return balanced_accuracy_score(data['Category'], category_predicted)

def evaluate_sentiment(data, sentiment_final):
  """
  Compute balanced accuracy on predicted sentiment.

  Args:
    data (pd.DataFrame): ground truth
    sentiment_final (list): predicted sentiment for articles
  """
  return balanced_accuracy_score(data['Sentiment'], sentiment_final)


def evaluate_incident(data, incident_prediction):
  """
  Compute metric on predicted incidents.

  Args:
    data (pd.DataFrame): ground truth
    incident_prediction (list): predicted incidents for articles
  """
  # Load a pretrained Sentence Transformer model
  model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', similarity_fn_name=SimilarityFunction.COSINE)

  #calculate embeddings for each article
  embeddings_incidents = model.encode(incident_prediction)
  embeddings_articles=["" for _ in range(len(data))]
  for i in range(len(data)):
    embeddings_articles[i]=model.encode(data[i])

  # Calculate the embedding similarities (cosine similarities, according to documentation)
  similarities_for_articles=["" for _ in range(len(embeddings_articles))]
  for i in range(len(embeddings_articles)):
    similarities_for_articles[i]=model.similarity(embeddings_incidents[i], embeddings_articles[i]).item()
  return similarities_for_articles, sum(similarities_for_articles)/len(similarities_for_articles)

def load_data_preprocessed(name):
  """
  Load one of the preprocessed datasets.

  Args: 
    name (str): name of the dataset which shuld be loaded.
  """
  df_gt=pd.read_csv(PATH_GT_DATA, sep=';')

  if name=="not_preprocessed_data":
    with open(PATH_NOT_PREPROCESSED, "r") as f:
        data_df = f.readlines()
  elif name=="processed_4o_data":
    with open(PATH_4O, "r") as f:
      data_df = f.readlines()
  elif name=="processed_regex_data":
    with open(PATH_REGEX, "r") as f:
      data_df = f.readlines()
  elif name =="processed_human_data":
    with open(PATH_HUMAN, "r") as f:
      data_df = f.readlines()
  else:
    raise ValueError("Wrong dataset name")
  return df_gt, data_df

def get_final_sentiment(articles):
  """
  Get final sentiment for ensemble model (RoBERTa and graph-based) for given articles.

  Args:
    articles (list): list of articles which should be analysed.
  """
  sentiment_roberta=run_roberta_sentiment(articles)
  sentiment_graph=run_actaware_preprocessed(articles)
  sentiment_final=[sentiment_graph[i] if sentiment_graph[i]==sentiment_roberta[i] else 'neutral' for i in range(len(sentiment_graph))]
  return sentiment_final