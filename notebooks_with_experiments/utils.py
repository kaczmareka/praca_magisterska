import pandas as pd

from config_experiments import(
  PATH_GT_DATA,
  PATH_NOT_PREPROCESSED,
  PATH_4O,
  PATH_REGEX,
  PATH_HUMAN
)

def load_data_preprocessed(name):
  """
  Load one of the preprocessed datasets.

  Args: 
    name (str): name of the dataset which shuld be loaded.
  """

  if name=="gt":
    data_df=pd.read_csv(PATH_GT_DATA, sep=';')
  elif name=="not_preprocessed_data":
    with open(PATH_NOT_PREPROCESSED, "r") as f:
        data_df = f.readlines()
  elif name=="processed_4o_data": #articles_cleaned_by_4o
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
  return data_df