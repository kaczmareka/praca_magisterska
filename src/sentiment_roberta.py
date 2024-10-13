#sentiment-roberta
# import pandas as pd
# from sklearn.metrics import balanced_accuracy_score
# import numpy as np
import torch
# import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#roberta
def run_roberta_sentiment(articles):
  answers=[" " for _ in range(len(articles))]
  model_name="textattack/roberta-base-SST-2"
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  for i in range(len(articles)):
    input_ids = tokenizer.encode(articles[i], return_tensors="pt", max_length=512, truncation=True)
    input_ids = input_ids.to(device)
    model = model.to(device)
    output=model(input_ids)
    predictions = output.logits.argmax().item()
    sentiment_lab=['negative', 'positive']
    answers[i]=sentiment_lab[predictions]
  return answers