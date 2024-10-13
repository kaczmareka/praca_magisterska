import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def run_roberta_sentiment(articles):
  """
  Run RoBERTa model on given articles.

  Args:
    articles (list): list of articles on which the sentiment should be analysed.
  """
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