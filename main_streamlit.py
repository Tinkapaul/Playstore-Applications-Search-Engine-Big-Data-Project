import streamlit as st
import numpy as np 
import pandas as pd
import txtai
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def load_data_and_embedding():
    df = pd.read_csv('googleplaystore.csv')
    titles = df.dropna().App.values

    embeddings = txtai.Embeddings({
        'path':'sentence-transformers/all-MiniLM-L6-v2'
    })
    embeddings.load('embeddings.tar.gz')

    return df, titles, embeddings

df, titles, embeddings = st.cache_data(load_data_and_embedding)()

st.title('Google Play Store Search Engine')

query = st.text_input('Enter a search: ','')

# Filter options
rating_filter = st.slider('Select Minimum Rating:', min_value=0.0, max_value=5.0, step=0.1)
type_filter = st.selectbox('Select Type:', ['Free', 'Paid'])

if st.button('Search'):
    if query:
        result = embeddings.search(query, 5)

        actual_results = [(titles[x[0]], df.loc[x[0], 'Category'], df.loc[x[0], 'Rating']) for x in result if df.loc[x[0], 'Rating'] >= rating_filter and df.loc[x[0], 'Type'] == type_filter]

        for res in actual_results:
            st.write(f"App: {res[0]}, Category: {res[1]}, Rating: {res[2]}")
    else:
        st.write('Please enter a query')
