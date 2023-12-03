import pycountry
import requests
#from api import apikey
import streamlit as st

st.title('Search News')

col1,col2=st.columns([3,1])
with col1:
    topic=st.text_input('Enter a search term')

with col2:
    category=st.radio('Choose a news category:', ('Crypto', 'Stocks', 'Market'))
    btn=st.button('Search')

if btn:
   # country=pycountry.countries.get(name=user).alpha_2
   # url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey=aec8df23005442f7bbd3c1ee298dbe31"
    url=f"https://newsapi.org/v2/everything?q={topic}&apiKey=aec8df23005442f7bbd3c1ee298dbe31"
    r=requests.get(url)
    r=r.json()
    articles=r['articles']
    for article in articles:
        st.header(article['title'])
        st.write("Published At: ",article['publishedAt'])
        if article['author']:
            st.write(article['author'])
        st.write(article['source']['name'])
        st.write(article['description'])
        if article['urlToImage']:
            st.image(article['urlToImage'])
