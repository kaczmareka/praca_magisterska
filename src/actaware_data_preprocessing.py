import json
import pandas as pd
import emoji
import copy
from src.config import (
  PATH_ORIGINAL_DATA,
  LIST_OF_REGEX
)

def load_original_data(path_to_data=PATH_ORIGINAL_DATA, first_rows=100):
  """
  Loads the not preprocessed dataset from Actaware Inc. from given path. 
  User can provide how many first rows which will be loaded, so that the pipeline runs faster.

  Args:
    path_to_data (str): path to not preprocessed dataset from Actaware Inc., default: PATH_ORIGINAL_DATA
    first_rows (int): how many of first rows will be loaded, default: 100.
  """
  with open(path_to_data) as json_file:
    data = json.load(json_file)
  data_df = pd.DataFrame(data)[:first_rows]
  df_pr_steps=pd.DataFrame(list(data_df['ProcessingSteps']))
  df_raw= pd.DataFrame({'TitleRaw': df_pr_steps['TitleRaw'], 'DescriptionRaw': df_pr_steps['DescriptionRaw'], 'ContentRaw': df_pr_steps['ContentRaw']})
  data_df_small=data_df[['Title', 'Keywords', 'MatchedCompanies', 'NERs']]
  data_df_small_all=data_df_small.join(df_raw, how='left')
  data_df_small_all=data_df_small_all.drop(['Title'], axis=1)

  #regex cleaning
  #replace artefacts in ContentRaw with ''

  # to_replace1='Get the free Morning Headlines email for news from our reporters across the world'
  # to_replace2='Sign up to our free Morning Headlines'
  # to_replace3='email Please enter a valid email address Please enter a valid email address SIGN UP'
  # to_replace4='I would like to be emailed about offers, events and updates from The Independent.'
  # to_replace5='Read our privacy notice'
  # to_replace6='Thanks for signing up to the Morning Headlines email'
  # to_replace7='{{ #verifyErrors }}'
  # to_replace8='{{ message }}'
  # to_replace9='{{ /verifyErrors }}'
  # to_replace10='{{ ^verifyErrors }}'
  # to_replace11='Something went wrong. Please try again later'
  # to_replace12=r'Advertisement !- - ad:.*?- -> '
  # to_replace13=r'<!- - ad:.*?- ->'
  # to_replace14=r'<a href=.*?</a>'
  # to_replace15=r'http.*? '
  # to_replace16='CLICK HERE TO SUBSCRIBE TO FOX NATION'
  # to_replace17='CLICK HERE FOR MORE SPORTS COVERAGE ON FOXNEWS.COM'
  # to_replace18='CLICK HERE TO GET THE FOX NEWS APP'
  # to_replace19='CREATE FREE ACCOUNT'
  # to_replace20='CLICK HERE TO DONATE TO "MAKE CAMO YOUR CAUSE"'
  # to_replace21='CLICK HERE FOR MORE SPORTS COVERAGE ON FOXNEWS.COM'
  # to_replace22='subscribe to my free CyberGuy Report Newsletter by heading to Cyberguy.com/Newsletter CLICK HERE'

  # List of patterns to replace with ''
  to_replace = LIST_OF_REGEX

  # Replace patterns in 'ContentRaw'
  for pattern in to_replace:
      data_df_small_all['ContentRaw'] = data_df_small_all['ContentRaw'].str.replace(pattern, '', regex=bool('regex' in pattern))

  # Normalize whitespace
  data_df_small_all['ContentRaw'] = data_df_small_all['ContentRaw'].str.replace(r'\s+', ' ', regex=True)

  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace1, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace2, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace3, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace4, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace5, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace6, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace7, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace8, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace9, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace10, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace11, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace12, '', regex=True)
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace13, '', regex=True)
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace14, '', regex=True)
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace15, '', regex=True)
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace16, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace17, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace18, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace19, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace20, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace21, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(to_replace22, '')
  # data_df_small_all['ContentRaw']=data_df_small_all['ContentRaw'].str.replace(r'\s+', ' ', regex=True)

  

  # remove records which have `None` values in either TitleRaw or ContentRaw
  index_with_none=data_df_small_all[data_df_small_all['ContentRaw'].isnull() + data_df_small_all['TitleRaw'].isnull()].index
  data_df_small_no_none=data_df_small_all.drop(index_with_none).reset_index()
  data_df_small_no_none=data_df_small_no_none.drop(['index'], axis=1)

  # remove records which have empty list in `MatchedCompanies`
  data_df_small_no_empty = data_df_small_no_none[data_df_small_no_none['MatchedCompanies'].apply(len)>0]

  #remove emoji
  data_df_small_no_emoji=copy.deepcopy(data_df_small_no_empty)
  for i in range(data_df_small_no_emoji.shape[0]):
    title_raw=data_df_small_no_emoji['TitleRaw'][i]
    title_raw=title_raw.replace('\n\n', ' ')
    title_raw=title_raw.replace('\n', '')
    description_raw=data_df_small_no_emoji['DescriptionRaw'][i]
    description_raw=description_raw.replace('\n\n', ' ')
    description_raw=description_raw.replace('\n', '')
    content_raw=data_df_small_no_emoji['ContentRaw'][i]
    content_raw=content_raw.replace('\n\n', ' ')
    content_raw=content_raw.replace('\n', '')
    data_df_small_no_emoji['TitleRaw'][i]=emoji.replace_emoji(title_raw, '')
    data_df_small_no_emoji['DescriptionRaw'][i]=emoji.replace_emoji(description_raw, '')
    data_df_small_no_emoji['ContentRaw'][i]=emoji.replace_emoji(content_raw, '')

  final_data=data_df_small_no_emoji[['ContentRaw', 'MatchedCompanies']]

  return final_data