import streamlit as st
import streamlit.components.v1 as components
# import tensorflow as tf
# import tensorflow_hub as hub
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import plotly.express as px
import re
import string
import nltk
# nltk.download("stopwords")
# from nltk.corpus import stopwords
# import tokenization


st.set_page_config(layout="wide")

header = st.container()
dataset = st.container()
plots = st.container()
model_training = st.container()

## Early variable and calculations
# stop_words = set(stopwords.words("english"))
punctuations = string.punctuation
url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
hashtag_pattern = r"(?i)\B((\#[a-zA-Z]+))"
mention_pattern = r"(?i)\B((\@[a-zA-Z]+))"


## Processing functions
def find_specific_word(pattern, string):
    """
    A function that uses RegEx to find words (strings) that matches a specific pattern.

    Args:
      pattern: a raw string that has the pattern we are looking for.
      string: the string we want to check if it has the pattern.
      
    Returns:
      A list of strings that matches the pattern of interest.
    """
    words = re.findall(pattern=pattern, string=string)
    
    return [x[0] for x in words]

def get_top_count_vectorizer(sentence_list, ngram, n):
    """
    Arge:
      sentence_list: a list of sentences to fit the vectorizer on
      ngram: a tuple that has the upper and lower boundary of the range n-values
      n: the number of words to be returned
      
    Returns:
      Two pandas series of n words used mostly in both classes
    """
    
    # Convert text into matrix of token counts
    vectorizer = CountVectorizer(ngram_range=ngram, # The lower and upper boundary of the range 
    # of n-values for different word n-grams or char n-grams to be extracted. All values of n such such 
    # that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams, 
    # (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams. Only applies if analyzer is not 
    # callable.
                                 stop_words='english' # a built-in stop word list for English is used.
                                )
    
    bag_of_words = vectorizer.fit_transform(sentence_list) # Fit then transform the vectorizer on the data
    
    sort_vocab = sorted(vectorizer.vocabulary_.items())
    list_vocab = [word[0] for word in sort_vocab]
    df_vectorizer = pd.DataFrame(bag_of_words.todense(), columns=list_vocab)

    # Get the target column from the main dataframe
    df_vectorizer["target"] = df_train["target"]

    # Get the most used word for each target
    df_vectorizer_0 = df_vectorizer[df_vectorizer["target"]==0].drop('target', axis=1)
    target_0_top_count = df_vectorizer_0.sum().sort_values(ascending=False)

    df_vectorizer_1 = df_vectorizer[df_vectorizer["target"]==1].drop('target', axis=1)
    target_1_top_count = df_vectorizer_1.sum().sort_values(ascending=False)
    
    return target_0_top_count[:n], target_1_top_count[:n]

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


## Streamlit functions
# @st.cache
def load_data(path:str):
    # Load CSV file
    df = pd.read_csv(path)
    ## Feature Engineering ##
    # Charecters
    df["char_count"] = df["text"].apply(lambda x: len(x))
    df["punc_count"] = df["text"].apply(lambda x: len([punc for punc in x if punc in punctuations]))
    df["upper_case_count"] = df["text"].apply(lambda x: len([c for c in x if c.isupper()]))
    df["lower_case_count"] = df["text"].apply(lambda x: len([c for c in x if c.islower()]))
    # Words
    df["words_list"] = df["text"].str.split()
    df["word_count"] = df["words_list"].apply(lambda x: len(x))
    df["unique_words"] = df["words_list"].apply(lambda x: set(x))
    df["unique_word_count"] = df["unique_words"].apply(lambda x: len(x))
    df["mean_word_length"] = df["words_list"].apply(lambda x: np.mean([len(i) for i in x]))
    # Text patterns
#     df["stop_words_count"] = df["words_list"].apply(lambda x: len([word for word in x if word in stop_words]))
#     df["non_stop_words_count"] = df["words_list"].apply(lambda x: len([word for word in x if word not in stop_words]))
    df["urls"] = df["text"].apply(lambda x: find_specific_word(url_pattern, x))
    df["url_count"] = df["urls"].apply(lambda x: len(x))
    df["hashtags"] = df["text"].apply(lambda x: find_specific_word(hashtag_pattern, x))
    df["hashtag_count"] = df["hashtags"].apply(lambda x: len(x))
    df["mentions"] = df["text"].apply(lambda x: find_specific_word(mention_pattern, x))
    df["mention_count"] = df["mentions"].apply(lambda x: len(x))

    return df

####################################################################
with header:
    st.title("Explaing BERT Predictions with LIME")
    st.markdown("In this project I will ...")

   
####################################################################
with dataset:
    st.header("Tweets Classification")
        
    df_test = load_data(r'test.csv')
    df_train = load_data(r'train.csv')
    

    if st.checkbox("Show training data:"):
        data_load_state = st.text('Loading data...')
        st.dataframe(df_train)
        data_load_state.text('Loading data...done!')
    if st.checkbox("Show test data:"):
        data_load_state = st.text('Loading data...')
        st.dataframe(df_test)
        data_load_state.text('Loading data...done!')


####################################################################
with plots:
    st.header("Displaying Features")
    numeral_features = tuple(df_test.select_dtypes(include=np.number).columns)
    numeral_features = list(numeral_features)[1:]
    
    # Make the radio buttons appear horizontally rather than vertically
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;} </style>', unsafe_allow_html=True)
    feature = st.radio("**Choose a feature to display a histogram for:**", tuple(numeral_features))


    train_col, test_col = st.columns(2)
    with train_col:
        st.subheader("Training Data Set")
        fig = px.histogram(df_train, x=feature, marginal="box", color_discrete_sequence=['turquoise'])
        st.plotly_chart(fig, use_container_width=True)


    with test_col:
        st.subheader("Test Data Set")
        fig = px.histogram(df_test, x=feature, marginal="box", color_discrete_sequence=['indianred'])
        st.plotly_chart(fig, use_container_width=True)


#     st.subheader("N-Gram")
#     n_gram_left, n_gram_right = st.columns(2)
#     with n_gram_left:
#         n_gram_lower = st.slider(min_value=1, max_value=5, label="Lower Boundary")
#     with n_gram_right:
#         n_gram_upper = st.slider(min_value=1, max_value=3, label="Upper Boundary")

#     if n_gram_lower > n_gram_upper: 
#         components.html("""
#         <p style='color:red'>
#         The Upper Boundary should NOT be less than the lower limit. Setting it to be equal to the Lower Boundary.
#         </p>
#         """)
#         n_gram_upper = n_gram_lower

#     n_gram = (n_gram_lower, n_gram_upper)
#     n_gram_num = st.slider(min_value=5, max_value=25, label="How many ngrams do you want to see?")
#     target_0_top_count, target_1_top_count = get_top_count_vectorizer(df_train["text"], ngram=n_gram, n=n_gram_num)
    
#     n_gram_for_target = st.selectbox(
#         label="Choose the target for which you want to see the N-Gram:", 
#         options=["Not a disaster", "Disaster"]
#     )
#     if n_gram_for_target == "Target 0":
#         target = target_0_top_count
#     else:
#         target = target_1_top_count
        
#     st.plotly_chart(px.bar(target, 
#                     labels={"index": "Phrase", "value": "Count"}).update_layout(showlegend=False))
