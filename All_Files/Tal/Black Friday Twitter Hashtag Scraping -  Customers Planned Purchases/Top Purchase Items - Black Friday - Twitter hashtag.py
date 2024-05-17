#!/usr/bin/env python
# coding: utf-8

# In[87]:


# Imports
import tweepy
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

import csv
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# In[60]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
IS_LOCAL = True

import os

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors


from time import time 
from collections import defaultdict

import plotly
import plotly.express as px
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode, plot
cf.go_offline()


# In[4]:


# Auth keys -  create twitter developer account to get auth keys

consumer_key = 
consumer_secret = 
access_token = 
access_token_secret = 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)


# In[56]:


# get username and tweet id - where the takealot black friday hashtag is
name = 'bellz_motshwane'
tweet_id = '1456179071646306306'

replies=[]
for tweet in tweepy.Cursor(api.search_tweets,q='to:'+name, result_type='recent', timeout=999999).items(1000):
    if hasattr(tweet, 'in_reply_to_status_id_str'):
        if (tweet.in_reply_to_status_id_str==tweet_id):
            replies.append(tweet)

with open('replies_clean.csv', 'w',encoding='utf-8') as f:
    csv_writer = csv.DictWriter(f, fieldnames=('user', 'text'))
    csv_writer.writeheader()
    for tweet in replies:
        row = {'user': tweet.user.screen_name, 'text': tweet.text.replace('\n', ' ')}
        csv_writer.writerow(row)


# In[113]:


df.head()


# In[61]:


df['text'] = df.text.str.replace('@bellz_motshwane', '')


# In[64]:


df.shape


# In[65]:


import re

# Define a function to clean the text
def clean(text):
    # Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', str(text))
    return text

# Cleaning the text in the review column
df['Cleaned_Text'] = df['text'].apply(clean)
df.head()


# In[66]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet

# POS tagger dictionary
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

df['POS_tagged'] = df['Cleaned_Text'].apply(token_stop_pos)
df.head()


# In[67]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos: 
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:  
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew
    
df['Lemma'] = df['POS_tagged'].apply(lemmatize)
df.head()


# In[83]:


Counter(" ".join(df["Lemma"]).split()).most_common(100)


# In[80]:


#convert the list above to dataframe
data = Counter(" ".join(df["Lemma"]).split()).most_common(100)


# In[81]:


data = pd.DataFrame(data,columns = ['text','frequency'])


# In[82]:


data.head()


# In[106]:


data.to_csv('tweet_data.csv')


# In[109]:



top_purchase_items = tweet_data.sort_values(by='frequency',ascending=False)[:30]



plt.figure(figsize=(16,6))
sns.barplot(x='text', y='frequency', data=top_purchase_items)
plt.title('Top Purchase Items', fontsize=20)
plt.ylabel('Count')
plt.ticklabel_format(style='plain', axis='y'); # set scientific notation off
plt.xticks(rotation=30, ha='right',fontsize=12)
plt.show()


# In[111]:


# word cloud to generate most occuring word
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

text = tweet_data.text

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(str(text))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




