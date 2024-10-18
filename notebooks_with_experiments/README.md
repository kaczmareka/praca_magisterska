# Notebooks used to conduct experiments

The notebooks present in this directory were used to conduct experiments described in the thesis. To run each of them the environment `env_actaware` is needed. Additionally, for some of them exist other requirements listed below or in `requirements.txt` file in this directory. Some of the files needed GPU, therefore were run sepatarely on GoogleColab. Some of the notebooks require additional datasets or files.

## Notebooks
* `actaware_data_preprocessing.ipynb`
    * *Content:* preprocessing of the Actaware Inc. dataset.
    * *Path:* exception, is present in the main directory.
    * *Input:* To run it, file `data/Articles_2023.json` is needed. 
    * *Output:* Creates files `data/list_of_contents_new.txt` and `data/data_df_small_no_emoji_without_http.csv`.

* `data_regex_preprocessing_chosen_data.ipynb`
    * *Content:* preprocessing of the subset of Actaware Inc. dataset using regex.
    * *Input:* To run it, file `data/chosen_articles.txt` is needed. The content of this file was chosen manually.
    * *Output:* Creates file `data/chosen_articles_cleaned_regex.txt`.

* `data_gpt_4o_mini_preprocessing_chosen_data.ipynb`
    * *Content:* preprocessing of the subset of Actaware Inc. dataset using GPT-4o-mini.
    * *Input:* To run it, file `data/chosen_articles.txt` is needed. The content of this file was chosen manually.
    * *Output:* Creates file `data/chosen_articles_cleaned_4o.txt`.

* `gpt_experiments_and_evaluation.ipynb`
    * *Content:* all experiments conducted with the use of GPT-3.5-turbo and GPT-4o-mini models.
    * *Input:* To run it, the following files are neccessary: `data/articles_categories_their_matched_companies.csv`, `data/chosen_articles.txt`, `data/chosen_articles_cleaned_4o.txt`, `data/chosen_articles_cleaned_regex.txt`, `data/chosen_articles_cleaned_by_me.txt`.
    * *Output:* results for all experiments have to be analyzed manually, however, they are saved in files in directory `results`.

* `count_scores_for_sentiment_category.ipynb`
    * *Content:* computing scores for all experiments achieved with the use of GPT-3.5-turbo and GPT-4o-mini models.
    * *Input:* To run it, the results of the experiments are needed, saved properly.
    * *Output:* results for all experiments have to be analyzed manually, however, they are saved in files in directory `results`.

* `sentiment_graph_based_with_experiments.ipynb`
    * *Content:* notebook with implementation and experiments using graph-based solution.
    * *Input:* To run all steps in it, twitter sentiment extraction, news sentiment and all subsets of Actaware Inc. are needed.
    * *Output:* results for all experiments have to be analyzed manually, however, they are saved in files in directory `results`.

* `bertopic_final_experiments.ipynb`
    * *Content:* notebook with model and metrics, measuing the final scores.
    * *Input:* To run this notebook, access to good computational resources are recommended, including GPU. Also, file `list_of_contents.txt` or similar is needed.
    * *Output:* results for all experiments have to be analyzed manually.

* `bert_roberta_ensembles_sentiment.ipynb`
    * *Content:* notebook with BERT, RoBERTa and ensembles, with evaluation.
    * *Input:* To run this notebook, access to good computational resources are recommended, including GPU. Also, results from graph-based solution are needed for the ensemble part.
    * *Output:* results for all experiments have to be analyzed manually, some of them are saved in specified directory.

* `KeyBERT_with_keywords_scores.ipynb`
    * *Content:* notebook with  KeyBERT experiments with evaluation.
    * *Input:* To run this notebook, access to good computational resources are recommended, including GPU. Also, all versions of subsets need to be stored in predefined directory.
    * *Output:* results for all experiments have to be analyzed manually.