import string
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

## Early variable and calculations
# stop_words = set(stopwords.words("english"))
punctuations = string.punctuation
url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s(" \
              r")<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])) "
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




def load_and_process_data(path: str):
    """
    Takes a path to a CSV file and preprocess the text data in it.

    Args:
      path: the path to a dataset
    
    Returns:
      A dataframe with text-related features
    """
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
    # df["urls"] = df["text"].apply(lambda x: find_specific_word(url_pattern, x))
    # df["url_count"] = df["urls"].apply(lambda x: len(x))
    # df["hashtags"] = df["text"].apply(lambda x: find_specific_word(hashtag_pattern, x))
    # df["hashtag_count"] = df["hashtags"].apply(lambda x: len(x))
    # df["mentions"] = df["text"].apply(lambda x: find_specific_word(mention_pattern, x))
    # df["mention_count"] = df["mentions"].apply(lambda x: len(x))

    return df





def get_top_count_vectorizer(df, sentence_list, ngram, n):
    """
    Arge:
      df: The training datafreme that contains target column
      sentence_list: a list of sentences to fit the vectorizer on
      ngram: a tuple that has the upper and lower boundary of the range n-values
      n: the number of words to be returned
      
    Returns:
      Two pandas series of n words used mostly in both classes
    """

    # Convert text into matrix of token counts
    vectorizer = CountVectorizer(ngram_range=ngram,  # The lower and upper boundary of the range
                                 # of n-values for different word n-grams or char n-grams to be extracted. All values of n such such
                                 # that min_n <= n <= max_n will be used. For example an ngram_range of (1, 1) means only unigrams,
                                 # (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams. Only applies if analyzer is not
                                 # callable.
                                 stop_words='english'  # a built-in stop word list for English is used.
                                 )

    bag_of_words = vectorizer.fit_transform(sentence_list)  # Fit then transform the vectorizer on the data

    sort_vocab = sorted(vectorizer.vocabulary_.items())
    list_vocab = [word[0] for word in sort_vocab]
    df_vectorizer = pd.DataFrame(bag_of_words.todense(), columns=list_vocab)

    # Get the target column from the main dataframe
    df_vectorizer["target"] = df["target"]

    # Get the most used word for each target
    df_vectorizer_0 = df_vectorizer[df_vectorizer["target"] == 0].drop('target', axis=1)
    target_0_top_count = df_vectorizer_0.sum().sort_values(ascending=False)

    df_vectorizer_1 = df_vectorizer[df_vectorizer["target"] == 1].drop('target', axis=1)
    target_1_top_count = df_vectorizer_1.sum().sort_values(ascending=False)

    return target_0_top_count[:n], target_1_top_count[:n]
