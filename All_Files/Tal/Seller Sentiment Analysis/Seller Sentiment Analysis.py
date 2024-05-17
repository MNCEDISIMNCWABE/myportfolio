#!/usr/bin/env python
# coding: utf-8

# ## Seller Sentiment Analyis : OCT 2020 - SEP 2021

# ### Rule based Sentiment Analysis
# 

# using TextBlob, VADER, SentiWordNet

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi

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


# In[3]:


data = pd.read_excel(r'nps_score_data.xlsx')


# In[68]:


data.head()


# In[6]:


# shape of the data
data.shape


# In[56]:


# surveys from Oct-2020 to Sep-2021
print("The data has surveys from {} to {}".format(min(data['Survey Month']),
                                                 max(data['Survey Month'])))


# In[7]:


data.info()


# In[9]:


# counting unique respondents 
n = len(pd.unique(data['Respondent ID']))
  
print("No.of Respondents :", n)


# In[47]:


# calculate nps score

detractors = data[data.nps_cat == 'Detractor'].shape[0]
promoters = data[data.nps_cat == 'Promoter'].shape[0]
responders = data['Respondent ID'].count()
#data['NPS_score'] = (promoters/responders - detractors/responders)*100

nps_score = (promoters/responders - detractors/responders)*100
print(nps_score)


# In[11]:


# Create NPS Category

def nps_category(row):
    if row["nps_recommend"] <= 6:
        return "Detractor"
    if row["nps_recommend"] >= 9:
        return "Promoter"
    else:
        return "Neutral"

data = data.assign(nps_cat = data.apply(nps_category, axis=1))
data.head()


# In[23]:


nps_cat = data['nps_cat'].value_counts()
nps_cat_df = pd.DataFrame({'labels': nps_cat.index,'values': nps_cat.values})

nps_cat_df.iplot(kind='pie',labels='labels',values='values', 
                  title='NPS Category', hole = 0.6,color=['seagreen','red','orange'])


# In[133]:


# respondents by NPS sore
plt.figure(figsize=(10,4))
sns.displot(df['nps_recommend']);


# In[30]:


# Respondents by Survey Month 
plt.figure(figsize=(12,6))

resp_id = data.groupby(['Survey Month'])['Respondent ID'].count()
plot = [x for x, df in data.groupby('Survey Month')]

plt.plot(plot, resp_id)
plt.xlabel("Survey Month")
plt.ylabel("Number of Respondents")
plt.title("Respondents by Survey Month");


# In[27]:


# Respondents by Survey Month and NPS Category
respondent = pd.crosstab(data['Survey Month'], data['nps_cat'])
respondent.plot(kind='bar', stacked=False, figsize=(14,6))
plt.xlabel("Survey Month",fontsize=14)
plt.ylabel("Number of Respondents",fontsize=14)
plt.xticks(rotation=0)
plt.title("Number of Respondents by Survey Month and NPS Category",fontsize=16);


# In[54]:


# fill missing values of the "message" column with text provided on the "additional feedback" column
data.message = np.where(data.message.isnull(), data.additional_feedback, data.message)
data.head()


# In[57]:


data.isnull().sum().sort_values(ascending = False)/len(data)*100 


# In[59]:


# keep only rows where message is not null
df = data[data['message'].notna()]


# In[61]:


df.shape


# In[60]:


df.head()


# In[67]:


responders = data.groupby(['message'])['Respondent ID'].count()[:5]


plt.figure(figsize=(16,6))
sns.barplot(x='message', y=responders, data=df)
plt.title('GMV by Tsin (top 15) - June to August', fontsize=16)
plt.ylabel('GMV')
plt.ticklabel_format(style='plain', axis='y'); # set scientific notation off
plt.xticks(rotation=90, ha='right',fontsize=12)
plt.show()


# In[71]:


import re

# Define a function to clean the text
def clean(text):
    # Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', str(text))
    return text

# Cleaning the text in the review column
df['Cleaned_Message'] = df['message'].apply(clean)
df.head()


# In[72]:


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

df['POS_tagged'] = df['Cleaned_Message'].apply(token_stop_pos)
df.head()


# In[73]:


df.to_csv('dfffff.csv')


# In[74]:


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
# print(df['review'][239] + "\n" + df['Lemma'][239])


# ## Sentiment analysis using TextBlob

# In[78]:


from textblob import TextBlob

# function to calculate subjectivity 
def getSubjectivity(message):
    return TextBlob(message).sentiment.subjectivity

# function to calculate polarity
def getPolarity(message):
    return TextBlob(message).sentiment.polarity

# function to analyze the messages
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


# In[79]:


# fin_data['Subjectivity'] = fin_data['Lemma'].apply(getSubjectivity) 
df['Polarity'] = df['Lemma'].apply(getPolarity) 
df['Analysis'] = df['Polarity'].apply(analysis)
df.head()


# In[81]:


# count negative, positve and eutral sentiments
tb_counts = df.Analysis.value_counts()
tb_counts


# In[83]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

tb_count= df.Analysis.value_counts()
plt.figure(figsize=(10, 7))
plt.pie(tb_counts.values, labels = tb_counts.index, explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False);
# plt.legend()


# ## Sentiment analysis using VADER

# In[84]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# function to calculate vader sentiment  
def vadersentimentanalysis(message):
    vs = analyzer.polarity_scores(message)
    return vs['compound']

df['Vader_Sentiment'] = df['Lemma'].apply(vadersentimentanalysis)

# function to analyse 
def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound <= -0.5 :
        return 'Negative'
    else:
        return 'Neutral'
    
df['Vader_Analysis'] = df['Vader_Sentiment'].apply(vader_analysis)
df.head()


# In[85]:


vader_counts = df['Vader_Analysis'].value_counts()
vader_counts


# In[86]:


vader_counts= df['Vader_Analysis'].value_counts()
plt.figure(figsize=(10, 7))
plt.pie(vader_counts.values, labels = vader_counts.index, explode = (0.1, 0, 0), autopct='%1.1f%%', shadow=False);


# ## Sentiment Analysis using SentiWordNet

# In[98]:


nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn

def sentiwordnetanalysis(pos_data):
    sentiment = 0
    tokens_count = 0
    for word, pos in pos_data:
        if not pos:
            continue
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        if not lemma:
            continue
        
        synsets = wordnet.synsets(lemma, pos=pos)
        if not synsets:
            continue

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1
        # print(swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score())
    if sentiment>0:
        return "Positive"
    if sentiment==0:
        return "Neutral"
    else:
        return "Negative"

df['SWN_analysis'] = df['POS_tagged'].apply(sentiwordnetanalysis)
df.head()


# In[99]:


swn_counts= df['SWN_analysis'].value_counts()
swn_counts


# In[100]:


swn_counts= df['SWN_analysis'].value_counts()
plt.figure(figsize=(10, 7))
plt.pie(swn_counts.values, labels = swn_counts.index, explode = (0.1, 0, 0), autopct='%1.1f%%', shadow=False);


# ## Visual representation of TextBlob, VADER, SentiWordNet results

# In[101]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(15,7))
plt.subplot(1,3,1)
plt.title("TextBlob results")
plt.pie(tb_counts.values, labels = tb_counts.index, explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False)
plt.subplot(1,3,2)
plt.title("VADER results")
plt.pie(vader_counts.values, labels = vader_counts.index, explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False)
plt.subplot(1,3,3)
plt.title("SentiWordNet results")
plt.pie(swn_counts.values, labels = swn_counts.index, explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False);


# In[107]:


# scatterplot of NPS and sentiment scores (vader and textblob)
sns.pairplot(df, vars=['Polarity','nps_recommend','Vader_Sentiment'])
plt.show()


# In[127]:


# Checking for correlation between NPS and Polarity scores (sentiments)

corr_var = ['nps_recommend','Polarity','Vader_Sentiment']
corr_df = df.loc[:,corr_var]

plt.figure(figsize = (15,6))
sns.heatmap(corr_df.corr(), cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6);


# In[110]:


sns.lineplot(data=df, x="nps_recommend", y="Polarity");


# In[111]:


sns.lineplot(data=df, x="nps_recommend", y="Vader_Sentiment");


# In[119]:


sns.lmplot(x='nps_recommend',y='Polarity',hue='Analysis',data=df, palette='Set1',aspect=2)
plt.xlabel("NPS",fontsize=14)
plt.ylabel("Sentiment Scores",fontsize=14)
plt.xticks(rotation=0);


# In[120]:


sns.regplot(x="nps_recommend", y="Polarity", data=df)


# In[121]:


# The greater the F score value the higher the correlation will be.
from scipy import stats

F, p = stats.f_oneway(df[df.nps_cat=='Detractor'].Polarity,
                      df[df.nps_cat=='Neutral'].Polarity,
                      df[df.nps_cat=='Promoter'].Polarity)

print(F)


# In[149]:


# word cloud to generate most occuring word
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[152]:


# Start with one review:
text = df.message

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(str(text))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




