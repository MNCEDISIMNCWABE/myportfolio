#!/usr/bin/env python
# coding: utf-8

# In[33]:


import string
import re


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
#to preprocess tweets
import nltk
import string
import unicodedata
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

ps = nltk.PorterStemmer()
wn = WordNetLemmatizer()
stop = stopwords.words('english')
exclude = set(string.punctuation) 

words = nltk.download('words')

from bs4 import BeautifulSoup
from textblob import TextBlob

import contractions
import unicodedata
from unidecode import unidecode
from pandas_gbq import to_gbq
import os
from google.cloud import bigquery_storage
from google.cloud import bigquery

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel





os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/dealsshaunrikhotso/Documents/bigQuery/ayb_gcs_credentials.json" 
# Initialize the BigQuery client
client = bigquery.Client()

# Define the SQL query
sql_query_last_game_played = """
WITH ranked_entries AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_date DESC) AS rank
  FROM `ayoba-183a7.analytics_dw.user_daily_games`
  WHERE event_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
)
SELECT *
FROM ranked_entries
WHERE rank = 1 
  AND game IS NOT NULL 
  AND game != '';
"""

# Define the SQL query
sql_query_all_available_games = """
SELECT distinct game_title,game_id FROM `ayoba-183a7.analytics_dw.dim_games` 
WHERE game_title IS NOT NULL AND game_title != '';
"""

def get_bigquery_data(client, sql_query):
    # Run the query
    query_job = client.query(sql_query)

    # Get the results as a DataFrame
    df = query_job.to_dataframe()

    # Display the DataFrame
    return df
    
    
df_user_last_game_played = get_bigquery_data(client, sql_query_last_game_played)
df_all_available_games = get_bigquery_data(client, sql_query_all_available_games)


class NltkPreprocessingSteps:
    
    
    
    def __init__(self, X):
        
        self.X = X


        self.sw_nltk = stopwords.words('english')
        new_stopwords = ['<*>','Ayoba','ayoba']
        self.sw_nltk.extend(new_stopwords)
        #self.sw_nltk.remove('not')
        '''
        self.pos_tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
        '''

        # '!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~' 32 punctuations in python
        # we dont want to replace . first time around
        self.remove_punctuations = string.punctuation.replace('.','')

    def remove_html_tags(self):
        self.X = self.X.apply(
                lambda x: BeautifulSoup(x, 'html.parser').get_text())
        return self


    def remove_accented_chars(self):
        self.X = self.X.apply(
                lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
        return self

    def replace_diacritics(self):
        self.X = self.X.apply(
                lambda x: unidecode(x, errors="preserve"))
        return self

    def to_lower(self):
        self.X = self.X.apply(lambda x: " ".join([word.lower() for word in x.split() if word and word not in self.sw_nltk]) if x else '')
        return self

    def expand_contractions(self):
        self.X = self.X.apply(
                lambda x: " ".join([contractions.fix(expanded_word) 
                            for expanded_word in x.split()]))
        return self

    def remove_numbers(self):
        self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
        return self
    
    def remove_http(self):
        self.X = self.X.apply(lambda x: re.sub(r'http\S+', '', x))
        return self
                                               
    def remove_words_with_numbers(self):
        self.X = self.X.apply(lambda x: re.sub(r'\w*\d\w*', '', x))
        return self
    
    def remove_digits(self):
        self.X = self.X.apply(lambda x: re.sub(r'[0-9]+', '', x))
        return self
    
    def remove_special_character(self):
        self.X = self.X.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]+', ' ', x))
        return self
    
    def remove_white_spaces(self):
        self.X = self.X.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        return self
    
    def remove_extra_newlines(self):
        self.X == self.X.apply(lambda x: re.sub(r'[\r|\n|\r\n]+', ' ', x))
        return self


    def replace_dots_with_spaces(self):
        self.X = self.X.apply(lambda x: re.sub("[.]", " ", x))
        return self

    def remove_punctuations_except_periods(self):
        self.X = self.X.apply(
                     lambda x: re.sub('[%s]' %
                      re.escape(self.remove_punctuations), '' , x))
        return self

    def remove_all_punctuations(self):
        self.X = self.X.apply(lambda x: re.sub('[%s]' %
                          re.escape(string.punctuation), '' , x))
        return self

    def remove_double_spaces(self):
        self.X = self.X.apply(lambda x: re.sub(' +', '  ', x))
        return self

    def fix_typos(self):
        self.X = self.X.apply(lambda x: str(TextBlob(x).correct()))
        return self

    def remove_stopwords(self):
        # remove stop words from token list in each colum
        self.X = self.X.apply(
            lambda x: " ".join([ word for word in x.split() 
                                if word not in self.sw_nltk]) )
        return self
    
    def remove_singleChar(self):
        # remove stop words from token list in each colum
        self.X = self.X.apply(
            lambda x: " ".join([ word for word in x.split() 
                                if len(word)>2]) )
        return self

    def lemmatize(self):
        lemmatizer = WordNetLemmatizer()
        self.X = self.X.apply(
            lambda x: " ".join([ wn.lemmatize(word) for word in x.split()]))
        
        return self

    def get_processed_text(self):
        return self.X
    
txt_preproc_all_games = NltkPreprocessingSteps(df_all_available_games['game_title'] )
txt_preproc_user_games = NltkPreprocessingSteps(df_user_last_game_played['game'] )

processed_text_all_games =   txt_preproc_all_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()


processed_text_all_users =   txt_preproc_user_games.to_lower().remove_html_tags().remove_accented_chars().replace_diacritics().expand_contractions().remove_numbers().remove_digits().remove_special_character().remove_white_spaces().remove_extra_newlines().replace_dots_with_spaces().remove_punctuations_except_periods().remove_words_with_numbers().remove_singleChar().remove_double_spaces().lemmatize().remove_stopwords().get_processed_text()

df_all_available_games['game_title_processed']=processed_text_all_games
df_user_last_game_played['game_processed']=processed_text_all_users


# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_all_available_games['game_title_processed'] )

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on content similarity
def get_content_based_recommendations(game_name, cosine_sim=cosine_sim, df_all_available_games=df_all_available_games):
    idx = df_all_available_games[df_all_available_games['game_title_processed'] == game_name].index
    if len(idx) == 0:
        print(f"No similar games found for '{game_name}'")
        return []
    else:
        idx = idx[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Top 10 similar games
        similar_games = [df_all_available_games['game_title_processed'][i[0]] for i in sim_scores]
        return similar_games


def get_recommendations():
    distinct_games = df_user_last_game_played['game_processed'].drop_duplicates().dropna()
    games_recommendations_df=''
    if distinct_games.empty:
        print("No distinct games found. Exiting recommendation process.")
    else:
        recommendations_list = []

        for game_name in distinct_games:

            recommendations = get_content_based_recommendations(game_name)
            print(f"Content-based recommendations similar to '{game_name}':")
            for rank, recommendation in enumerate(recommendations, start=1):
                recommendations_list.append({
                    'game_name': game_name,
                    'recommendation_game': recommendation,
                    'ranking': rank
                })

        # Create the DataFrame after the loop
        recommendations_df = pd.DataFrame(recommendations_list)

        # Display the DataFrame with recommendations for each game
        print("\nDataFrame with recommendations for each game:")
        games_recommendations_df =recommendations_df
        return games_recommendations_df  
    

def pre_process_recommedation(df= get_recommendations()):
    
    df_reco= df
    #get_recommendations()
    #df_reco = pd.merge( df_reco,)

    # Perform a left join
    df_merged_reco = pd.merge(df_reco, df_all_available_games, left_on='recommendation_game', right_on='game_title_processed', how='left')
    df_user_rec=df_user_last_game_played[['user_id','game_processed']]
    df_user_games_recommendations = pd.merge(df_user_rec, df_merged_reco, left_on='game_processed', right_on='game_name', how='left')
    #df_user_games_recommendations['game_id'] = df_user_games_recommendations['game_id'].astype(int)
    df_user_games_recommendations['ranking'].fillna(-1, inplace=True)
    df_user_games_recommendations['ranking'] = df_user_games_recommendations['ranking'].astype(int)
    df_user_games_recommendations['country']=''
    df_user_games_recommendations['city'] =''
    df_user_games_recommendations['recommendation_type']='games'
    df_user_games_recommendations['recommendation_activity']='user_activity'
    df_user_games_recommendations[['user_id','game_id','ranking','recommendation_type','recommendation_activity']]
    #This is rename to conform with the current production database, we have to create a new table for the activity recommendation later and update the api point

    df_user_games_recommendations.rename(columns={'game_id': 'GameID'}, inplace=True)
    df_user_games_recommendations.rename(columns={'ranking': 'City_Ranking'}, inplace=True)
    return df_user_games_recommendations[['user_id','country','city','GameID','City_Ranking','recommendation_type','recommendation_activity']]

#RECO DATA
df=pre_process_recommedation(df=get_recommendations())

def create_and_load_recommendations(df):
    # Define the table schema
    schema = [
        bigquery.SchemaField("user_id", "STRING"),
        bigquery.SchemaField("country", "STRING"),
        bigquery.SchemaField("city", "STRING"),
        bigquery.SchemaField("GameID", "INTEGER"),
        bigquery.SchemaField("City_Ranking", "INTEGER"),
        bigquery.SchemaField("recommendation_type", "STRING"),
        bigquery.SchemaField("recommendation_activity", "STRING"),
    ]

    # Define table ID and configuration
    dataset_id = 'ayoba-183a7.analytics_dw' 
    table_id = f'{dataset_id}.rec_games_recommendations_activity_staging'
    table = bigquery.Table(table_id, schema=schema)

    # Drop the table if it exists
    try:
        client.delete_table(table)
        print(f"Dropped table {table_id}")
    except:
        pass  # Ignore if the table doesn't exist


    # Create the table
    # Set clustering fields
    table.clustering_fields = ["user_id"]

    # Create the table
    table = client.create_table(table, exists_ok=True)
    print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}") 
    # Load the DataFrame into the newly created table
    to_gbq(df, table_id, project_id=table.project, if_exists='append')
    print('Data loading done')


create_and_load_recommendations(df)

def manage_down_stream_update():
    # Define the SQL for the MERGE operation
    merge_sql = """
    MERGE `ayoba-183a7.analytics_dw.rec_card_recommendations` AS A
    USING (SELECT * FROM `ayoba-183a7.analytics_dw.rec_games_recommendations_activity_staging` 
            where GameId is not null or GameId !=-1)  AS B
    ON (A.user_id = B.user_id 
        AND A.GameID = B.GameID
        AND B.recommendation_type = B.recommendation_type
        AND B.recommendation_activity)
    WHEN MATCHED AND A.ranking != B.ranking THEN
      UPDATE SET 

        A.ranking = B.ranking,

    WHEN NOT MATCHED BY TARGET THEN
      INSERT (user_id, country, city, GameID, City_Ranking, recommendation_type, recommendation_activity)
      VALUES (B.user_id, B.country, B.city, B.GameID, B.City_Ranking, B.recommendation_type, B.recommendation_activity)
    WHEN NOT MATCHED BY SOURCE 
        AND recommendation_type = 'user_activity' 
        AND recommendation_type = 'games' THEN
      DELETE
    """

    # Execute the SQL statement
    query_job = client.query(merge_sql)
    
manage_down_stream_update()


# In[ ]:





# In[ ]:



    query_job.result()  # Wait for the job to complete

