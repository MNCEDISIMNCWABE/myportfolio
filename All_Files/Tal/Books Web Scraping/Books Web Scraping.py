#!/usr/bin/env python
# coding: utf-8

# In[26]:


from bs4 import BeautifulSoup
import requests
import csv


Base_URL = "https://www.kevinrooke.com"
try:
    with open('Books_Recommendation.csv', 'w', newline='') as f:

        thewriter = csv.writer(f)
        header = ['Book Tittle','Author', "Recommended by" ]
        thewriter.writerow(header)

        source =requests.get(Base_URL+"/book-recommendations")
        source.raise_for_status()
        soup = BeautifulSoup(source.text,'html.parser')
        Profile_list= soup.find_all('div', class_="profile-item")
        for  li in Profile_list:
            profile_url=  Base_URL+li.a['href']
            RecommendedBy =  li.find('div', class_="profile-name").text
       
            profile_source =requests.get(profile_url)
            profile_source.raise_for_status()
            profile_soup = BeautifulSoup(profile_source.text,'html.parser')
            collection_list= profile_soup.find_all('div', class_="div-block-7")
            for item in collection_list:
                BookName= item.find('h1', class_="book-title").text
                Author= item.find('h2', class_="heading-9").text
                data= [BookName, Author,RecommendedBy ]
                print(data)
                thewriter.writerow(data)

except Exception as e:
    print(e)   


# In[ ]:




