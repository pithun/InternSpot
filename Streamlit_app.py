# Imports

# Data Manipulation and Visualization
import pandas as pd
import numpy as np
from PIL import Image

# Streamlit Modules
import streamlit as st
from st_card_component import card_component

# NLP Modules
from sentence_transformers import SentenceTransformer, util
from data_cleaning import prepare_document, cos_dicts
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import nltk

doc = 'punkt'

#@st.cache previously incase site misbehaves
@st.cache_resource
def load_nltk(doc):
    return nltk.download(doc)

load_nltk(doc)

# Images used in Page
image = Image.open('Images/LOGO.png')
image1 = Image.open('Images/count_states.jpg')
image2 = Image.open('Images/count_paid.jpg')

# Inputting the logo
st.image(image)
st.title('InternSpot (Beta Version)')

st.write('This is a repository containing RECENT Industrial training companies sourced directly from the'
         ' past set, the aim is to provide feasible companies to Students.')
st.write(' This app uses State of the art Machine Learning to help students search descriptively for the kind of Company they want. '
         'Who says you don\'t have a choice ??')
st.write('USE THE FILTERS or make an Advanced Search 🔍')

# Reading the data
files = pd.read_csv('IT_companies_data/Company_data.csv')

import streamlit as st

st.subheader('Incentive😁!!!')
st.write('For the Final years in Civil Engineering we provide;'
         + '\n' + '1. A [Google Drive](https://drive.google.com/drive/folders/1DieblSG7sz9OA3b_kJaQ3Y6CPGZOLmJe?usp=drive_link)'
                  ' with Rich content on all courses you\'ll be studying this year (2021/2022 set notes, past questions, '
                  'useful youtube links, and more to come!'
         + '\n' + '2. A way to help those coming after you we believe Internsp🔍t will play a huge role going forward and'
                  'it\'s only going to be possible through you'
         )

st.subheader('DataSet Overview')

# Using Streamlit's column function to Place two other Summary pictures in columns
col1, col2 = st.columns(2)

with col1:
    st.image(image1)
    st.write('There\'s currently a total of ' + str(len(files)) + ' Companies in our Dataset across 10 States'
                                                                  ' with majority ' \
                                                                  'in Lagos, stay tuned for more! 😁')
with col2:
    st.image(image2)
    st.write('We see the Paid-Non Paid fraction across States. All IT companies in Lagos were paid although '
             'sources didn\'t drop the amount 😅.')


st.subheader('SP🔍T A COMPANY')

#st.write('Use Filters or Make an Advanced search')

# Using Streamlit's column function to Place two other Summary pictures in columns
cola_1, cola_2 = st.columns(2)

with cola_1:
    # Creating the option for all States in the dataset
    states = ['All', 'Abia', 'Abuja', 'Anambra', 'Delta', 'Enugu', 'Lagos', 'Rivers', 'Ebonyi', 'Cross River',
              'Ogun']
    option_state = st.selectbox('What\'s your prefered state ?', options=states)

with cola_2:
    # Creating a multiselect to enable paid/not paid IT filtering
    options = st.multiselect(
        'Paid, Not Paid',
        ['Paid', 'Not Paid'], 'Paid')
show_table = st.checkbox('Show table instead')

# Creating input box for advances searches
to_be_searched = st.text_input('Advanced Search', placeholder='Describe your target company Instead')

# We used card below with three cards in a row, this calculations were wrt that
amount_of_cols = len(files) / 3
ceiled_amount = int(np.ceil(amount_of_cols))

# Here, if the option for paid is both paid and unpaid, we want to return the entire dataframe, else we return
# only a subset or. We used the try - except block to avoid error in a case where a user removes all filters
if len(options) == 2:
    files = files
else:
    try:
        files = files[files.Status == options[0]]
    except:
        pass

# Here basically, we limit the output based on users input State
if option_state == 'All':
    files = files
else:
    files = files[files.State == option_state]

# Applying Machine Learning
cleaned_about = files.About.apply(lambda x: prepare_document(x))
files['cleaned_about'] = cleaned_about

tfidf = TfidfVectorizer()
tfidf_about = tfidf.fit_transform(cleaned_about.tolist())
tfidf_about = tfidf.fit_transform(cleaned_about.tolist())
about_cos_dict = cos_dicts(files.Company, tfidf_about.toarray())
all_data = pd.read_csv('IT_companies_data/Company_data.csv')


@st.cache_resource
def embed(data):
    embedder = SentenceTransformer('msmarco-MiniLM-L6-cos-v5')
    embedding = embedder.encode(data.About.tolist(), convert_to_tensor=True)
    return embedder, embedding


about_embedding = embed(all_data)[1]
embedder = embed(all_data)[0]


@st.cache_resource
def nli_search(query):
    # given a query, return top few similar games

    # example code taken from Sentence Transformers docs
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, about_embedding)[0]
    top_results = torch.topk(cos_scores, k=5)

    ret_list = []

    for score, idx in zip(top_results[0], top_results[1]):
        ret_list.append((all_data.Company.tolist()[idx]))

    return ret_list


if st.button('Search') and len(to_be_searched) > 0:
    nat_lan_df = all_data.copy()
    nat_lan_df = nat_lan_df[nat_lan_df['Company'].isin(nli_search(to_be_searched))]
    nat_lan_df.reset_index(drop=True, inplace=True)
    st.write('NOTE: The results displays the top 5 most similar companies to your search')
    st.dataframe(nat_lan_df[['Company', 'Sector', 'Status', 'LinkedIn']])
else:
    # The show_table part returns a dataframe if the user prefers that view.
    st.markdown("""**Note:** :red[Advanced Search overwrites the result of the filters]""")
    if show_table:
        df = files.copy()[['Company', 'Sector', 'Status', 'LinkedIn']]
        df.reset_index(drop=True, inplace=True)
        st.write('Total of ' + str(len(files)) + ' Companies see them below')
        st.dataframe(df)
    else:
        # This part loops through the ceiled amount which was the rounding up result of dividing total companies
        # by 3. Basically, this helps us know how many rows of columns we're creating.
        try:
            st.write('Based on your filter, there\'s a total of ' + str(len(files)) + ' Companies see them below')

            subset_start = 0
            subset_end = 3
            key1 = 1
            key2 = 2
            key3 = 3
            for a in range(ceiled_amount):
                # st.write(a)
                subset = files.iloc[subset_start:subset_end, :]
                # st.write(subset.Company.values[0])
                colu1, colu2, colu3 = st.columns(3)
                with colu1:
                    card1 = card_component(title=subset.Company.values[0],
                                           context=subset.Sector.values[0],
                                           highlight_start=0,
                                           highlight_end=len(subset.Sector.values[0]),
                                           score=subset.Status.values[0] + ' IT - ' + subset.State.values[0],
                                           url=subset.LinkedIn.values[0],
                                           key=key1
                                           )
                with colu2:
                    card2 = card_component(title=subset.Company.values[1],
                                           context=subset.Sector.values[1],
                                           highlight_start=0,
                                           highlight_end=len(subset.Sector.values[1]),
                                           score=subset.Status.values[1] + ' IT - ' + subset.State.values[1],
                                           url=subset.LinkedIn.values[1],
                                           key=key2
                                           )
                with colu3:
                    card3 = card_component(title=subset.Company.values[2],
                                           context=subset.Sector.values[2],
                                           highlight_start=0,
                                           highlight_end=len(subset.Sector.values[2]),
                                           score=subset.Status.values[2] + ' IT - ' + subset.State.values[2],
                                           url=subset.LinkedIn.values[2],
                                           key=key3
                                           )
                key1 += 0.33
                key2 += 0.33
                key3 += 0.33

                subset_start += 3
                subset_end += 3

        except:
            pass

# st.subheader('Advanced Search Result')
# st.write('Hint: Select All States for our advanced search to go through the entire Dataset.')
# if st.sidebar.button('Search'):
#   nat_lan_df = files.copy()
#  nat_lan_df = nat_lan_df[nat_lan_df['Company'].isin(nli_search(to_be_searched))]
# nat_lan_df.reset_index(drop=True, inplace=True)
# st.dataframe(nat_lan_df[['Company', 'Sector', 'Status', 'LinkedIn']])
