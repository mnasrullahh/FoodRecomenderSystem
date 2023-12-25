#streamlit library
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from PIL import Image
import re
import time 

#visualization library
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from plotly.subplots import make_subplots
#import matplotlib 

#data manipulation library
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

ic = Image.open("icon.png")
st.set_page_config(
    page_title="Food Recommendation System",
    page_icon=ic,
    layout="wide"
)

#css file
with open('style.css')as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)
 
with st.sidebar:
    Food = Image.open(r'FDC-Logo.png')
    st.image(Food, width=250)
    st.markdown('\n')
    selected = option_menu("Main Menu", ["Dashboard", 'Application'], 
        icons=['clipboard-data', 'gear'], menu_icon="cast", default_index=0)
    st.write("Streamlit App by :")
    st.write("[Moch. Nasrullah Hasani](https://www.linkedin.com/in/moch-nasrullah-hasani/)")
#load data
@st.cache_data()
def load_data(url):
    data = pd.read_csv(url)
    return data

data = load_data('data_sample.csv')
#similarity = pd.read_csv('')

if selected=="Dashboard":
    st.header('Food Recommendation Dashboard')

    # Metrics Section
    with st.container():
        st.markdown('')

    m1, dum1, m2, dum2, m3, dum3, m4, dum4 = st.columns([2, 1, 2, 1, 2, 1, 3, 5])
    # Number of Recipes
    with m1:
        number_of_recipes = len(data)
        st.metric(label="Recipes", value=number_of_recipes)

    # Average TotalTime
    with m2:
        average_total_time = int(data['TotalTime'].mean())
        st.metric(label="Average Total Time", value=average_total_time)

    # Average Rating_avg
    with m3:
        average_rating = round(data['Rating_avg'].mean(), 2)
        st.metric(label="Average Rating", value=average_rating)

    # Average Sentiment Score
    with m4:
        average_sentiment_score = round(data['Sentiment_score'].mean(), 2)
        st.metric(label="Average Sentiment Score", value=average_sentiment_score)

#    st.subheader('WordCloud for Recipe Names')
 #   dum, word_cloud, dum = st.columns([1,9,1])
  #  with word_cloud:
   #     wordcloud = WordCloud(width=1000, height=500, background_color='white', colormap='viridis').generate(' '.join(data['Name']))
    #    plt.figure(figsize=(10, 6))
     #   plt.imshow(wordcloud, interpolation='bilinear') 
      #  plt.axis('off')
       # st.pyplot(plt)
  
    st.subheader("Recipe Data")
    dum1, recipe_data, dum2, review_count = st.columns([1,3,2,7])
    with recipe_data:
        fig = go.Figure(data=[go.Table(
            columnwidth = [3,1],
            header=dict(values=list(data[['Name','TotalTime']].columns),
                        fill_color='#F58A00',
                        align='center',
                        font=dict(color='Black', size=13)),
            cells=dict(values=[data.Name, data.TotalTime],
                        fill_color='#fce6c7',
                        align='left'))])
        fig.update_layout(margin=dict(l=2,r=2,b=5,t=5), height=400, width=400)
        st.write(fig)
        
    with review_count:
        # Sorting berdasarkan ReviewCount secara descending
        data_sorted = data.sort_values(by='ReviewCount', ascending=False)
        # Hanya mengambil 10 data teratas
        data_top10 = data_sorted.head(10).sort_values(by='ReviewCount', ascending=True)
        fig = go.Figure(go.Bar(
                                    x=data_top10.ReviewCount,
                                    y=data_top10.Name,
                                    orientation='h',
                                    marker_color="#e8962f"))
        fig.update_layout(title={ 
                                    'text': "Most Reviewed Recipe",
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'},plot_bgcolor='#ffffff')
        #fig.update_yaxes(visible=False)
        fig.update_xaxes(title='Count')
        fig.update_yaxes(title='Recipe')
        fig.layout.xaxis.fixedrange = True
        fig.layout.yaxis.fixedrange = True
        st.plotly_chart(fig,use_container_width=True)        


    st.subheader("Recipe Scoring Distribution")
    hist_rating, sentiment_score = st.columns([1,1])
    with hist_rating:
        fig = px.histogram(data, x="Rating_avg",
                    title='Rating Distribution',
                    opacity=0.9,
                    color_discrete_sequence=['#F58A00'],
                    nbins=50
                    )
        fig.update_layout(title={
                    'y':0.9,
                    'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},plot_bgcolor='#ffffff', showlegend=False)
        fig.update_xaxes(title='Rating')
        fig.update_yaxes(title='Count')
        fig.layout.xaxis.fixedrange = True
        fig.layout.yaxis.fixedrange = True
        st.plotly_chart(fig,use_container_width=True)

    with sentiment_score:
        fig = px.histogram(data, x="Sentiment_score",
                    title='Sentiment Score Distribution',
                    opacity=0.9,
                    color_discrete_sequence=['#F58A00'],
                    nbins=50
                    )
        fig.update_layout(title={
                    'y':0.9,
                    'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},plot_bgcolor='#ffffff', showlegend=False)
        fig.update_xaxes(title='Score')
        fig.update_yaxes(title='Count')
        fig.layout.xaxis.fixedrange = True
        fig.layout.yaxis.fixedrange = True
        st.plotly_chart(fig,use_container_width=True)

    cslid, slid,dum4 = st.columns([3,2,4])
    with cslid:
        rate_opt = st.radio("Rating Options:",['Greater than','Less than', 'Equal'], horizontal=True)#,label_visibility="hidden")
    with slid:
        rate_slid = st.slider("Rating:", min_value=0.0,max_value=5.0,value=0.0,step=0.1)

    time_dist, wc = st.columns([1,1])

    with time_dist:
        if rate_opt=="Greater than":
            selected = data[data['Rating_avg'] >= rate_slid]
        elif rate_opt=="Less than":
            selected = data[data['Rating_avg'] <= rate_slid]
        else:
            selected = data[data['Rating_avg'] == rate_slid]    

        # Total Time Distribution
        filtered_data = selected[selected['TotalTime'] <= 1500]
        fig = px.histogram(filtered_data, x="TotalTime",
                    title='Total Time Distribution',
                    opacity=0.9,
                    color_discrete_sequence=['#F58A00'],
                    nbins=100
                    )
        fig.update_layout(title={
                    'y':0.9,
                    'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},plot_bgcolor='#ffffff', showlegend=False)
        fig.update_xaxes(title='Total Time')
        fig.update_yaxes(title='Count')
        fig.layout.xaxis.fixedrange = True
        fig.layout.yaxis.fixedrange = True
        st.plotly_chart(fig,use_container_width=True)

    with wc:
        if rate_opt=="Greater than":
            selected = data[data['Rating_avg'] >= rate_slid]
        elif rate_opt=="Less than":
            selected = data[data['Rating_avg'] <= rate_slid]
        else:
            selected = data[data['Rating_avg'] == rate_slid]

        # reviews_text = ' '.join(selected['review_content'].dropna().values().strip().lower())
        stopwords = ['i','the','of','had','one','ve','yet','a','with','but','much','tot','to','your','kinda','my','family','all','ate','and','it','on','how','this','was','t','you','put','don','couldn','4oz','evelyn','in','is','that','very','make','just','1oz','got']
        reviews_text =[i.lower() for i in  re.split(r'\W+',' '.join(selected['Review'].dropna().values))]
        reviews_text = [i for i in reviews_text if i not in stopwords]

        def word_count(words):
            counts = dict()
            for word in words:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
            return counts
            
        wc = pd.Series(word_count(reviews_text)).head(15).sort_values(ascending=True)
        fig = go.Figure(go.Bar(
                                    y=wc.index,
                                    x=wc.values,
                                    orientation='h',
                                    marker_color="#F58A00"))
        fig.update_layout(title={ 
                                    'text': "Word Review Distribution",
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'},plot_bgcolor='#ffffff')
        #fig.update_yaxes(visible=False)
        fig.update_xaxes(title='Count')
        fig.update_yaxes(title='Word')
        fig.layout.xaxis.fixedrange = True
        fig.layout.yaxis.fixedrange = True
        st.plotly_chart(fig,use_container_width=True)


elif selected=="Application":
    st.header('Food Recommendation App')

    with st.container():
        col1, col2 = st.columns([2,3])
        with col1:
            st.markdown("#### Recipe Name")
            choose_recipe = st.selectbox("Recipe: ", set(data["Name"].to_list()), label_visibility='collapsed')

    # Tampilkan informasi dari resep yang dipilih
    st.subheader(f"Details for {choose_recipe}")
    
    m1, dum1, m2, dum2, m3, dum3, m4, dum4 = st.columns([3, 1, 3, 1, 2, 1, 2, 4])
    with m1:
        reviwed = data[data['Name'] == choose_recipe]['ReviewCount']
        st.metric(label="Get Reviewed", value=f"{reviwed.values[0]} times")
    with m2:
        time = data[data['Name'] == choose_recipe]['TotalTime']
        st.metric(label="Cooking Time", value=f"{time.values[0]} minutes")
    with m3:
        rating = data[data['Name'] == choose_recipe]['Rating_avg']
        rating = round(rating, 2)
        st.metric(label="Rating Average", value=rating)
    with m4:
        sentiment = data[data['Name'] == choose_recipe]['Sentiment_score']
        sentiment = round(sentiment, 2)
        st.metric(label="Sentiment Score", value=sentiment)

    tab1, tab2 = st.tabs(["Ingredient", "Instruction"])
    with tab1:
        ingredient = data[data['Name'] == choose_recipe]['RecipeIngredientParts']
        st.write(ingredient)
    with tab2:
        instruction = data[data['Name'] == choose_recipe]['RecipeInstructions']
        st.write(instruction)
    
    fig = go.Figure(data=[go.Table(
        columnwidth = [1,1],
        header=dict(values=['Ingredient','Instructions'],
                    line_color='#F58A00', fill_color='#F58A00',
                    align='center',font=dict(color='Black', size=20)),
        cells=dict(values=[selected_recipe_info.RecipeIngredientParts, selected_recipe_info.RecipeInstructions],
                    line_color='white', fill_color='white',
                    align='left',font=dict(color='Black', size=18)))])
    fig.update_layout(margin=dict(l=2,r=2,b=5,t=5), height=200, width=1000, font_family="Times New Roman") #"Arial", "Balto", "Courier New", "Droid Sans",, "Droid Serif", "Droid Sans Mono", "Gravitas One", "Old Standard TT", "Open Sans", "Overpass", "PT Sans Narrow", "Raleway", "Times New Roman".
    st.write(fig) 

    st.subheader("Recipe Recommendation")
    col, dum = st.columns([3,7])
    with col:
        with st.spinner("Searching Recomendation Recipe..."):
            # Menggabungkan teks dari kolom 'RecipeIngredientParts' dan 'Name' menjadi satu kolom baru 'CombinedText'
            data['CombinedText'] = data['RecipeIngredientParts'].astype('str') + ' ' + data['RecipeInstructions'].astype('str')
            # Menggunakan TfidfVectorizer pada kolom 'CombinedText'
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(data['CombinedText'])

            # Menghitung cosine similarity
            similarity = cosine_similarity(tfidf_matrix)

            index = data[data['Name'] == choose_recipe].index[0]

            # Get the pairwise similarity scores of the product
            sim_scores = list(enumerate(similarity[index]))

            # Sort the products based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the top 10 similar products
            top_products = pd.DataFrame({'Recipe Name':[data.iloc[score[0]]['Name'] for score in sim_scores[1:11]],
                                        'Similarity Score':[score[1] for score in sim_scores[1:11]],
                                        'Rating Average':[data.iloc[score[0]]['Rating_avg'] for score in sim_scores[1:11]],
                                        'Sentiment Score':[data.iloc[score[0]]['Sentiment_score'] for score in sim_scores[1:11]],
                                        'Inggridients':[data.iloc[score[0]]['RecipeIngredientParts'] for score in sim_scores[1:11]],
                                        'Instructions':[data.iloc[score[0]]['RecipeInstructions'] for score in sim_scores[1:11]]
                                        }).sort_values('Similarity Score',ascending=False)
    st.dataframe(top_products, use_container_width=True)
