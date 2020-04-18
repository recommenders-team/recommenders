# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import BertTokenizer
import re, string, unicodedata
import pandas as pd
import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer

def clean_text(text, for_BERT=False, verbose=False):
    """ Clean text by removing HTML tags, symbols, and punctuation.
    
    Args:
        text (str): Text to clean.
        for_BERT (boolean): True or False for if this text is being cleaned for future BERT tokenization.
        verbose (boolean): True or False for whether to print.
    
    Returns:
        clean (str): Cleaned version of text.
    """
    
    try:
        # Normalize unicode
        text_norm = unicodedata.normalize('NFC', text)

        # Remove HTML tags
        clean = re.sub('<.*?>', '', text_norm)

        # Remove new line and tabs
        clean = clean.replace('\n', ' ')
        clean = clean.replace('\t', ' ')
        clean = clean.replace('\r', ' ')
        clean = clean.replace('Â\xa0', '')     # non-breaking space

        # Remove all punctuation and special characters
        clean = re.sub('([^\s\w]|_)+','', clean)

        # If you want to keep some punctuation, see below commented out example
        # clean = re.sub('([^\s\w\-\_\(\)]|_)+','', clean)

        # Skip further processing if the text will be used in BERT tokenization
        if for_BERT is False:
            # Lower case
            clean = clean.lower()
    except:
        if verbose is True:
            print('Cannot clean non-existent text')
        clean = ''
    
    return clean

def clean_dataframe_for_rec(df, cols_to_clean, for_BERT=False):
    """ Clean the text within the columns of interest and return a dataframe with cleaned and combined text.
    
    Args:
        df (pd.DataFrame): Dataframe to clean.
        cols_to_clean (str): List of columns to clean by name (e.g., ['title','abstract','full_text']).
        for_BERT (boolean): True or False for if this text is being cleaned for future BERT tokenization.
    
    Returns:
        df (pd.DataFrame): Dataframe with cleaned text in new column 'cleaned_text'.
    """
    # Clean the text in the dataframe
    for i in range(0, len(df)):
        for col in cols_to_clean:
            df[col][i] = clean_text(df[col][i], for_BERT)

    # Collapse the table such that all descriptive text is just in a single column
    first_col = True
    for col in cols_to_clean:
        if first_col is True:
            # first one
            df['cleaned_text'] = df[col]
            first_col = False
        else:
            df['cleaned_text'] = df['cleaned_text'] + ' ' + df[col]
    
    # Make sure any punctuation or special characters in Name are human readible
    for i in range(0, len(df)):
        df['title'][i] = df['title'][i].encode('ascii','ignore')
    
    for i in range(0, len(df)):
        df['title'][i] = df['title'][i].decode()
    
    return df

def tokenize_with_BERT(vector, bert_method='bert-base-cased'):
    """ Tokenize the input text with HuggingFace BERT tokenization.
    
    Args:
        vector (pd.Series): Series of textual descriptions (typically values from pandas series df_clean['full_text']).
    
    Returns:
        vector_tokenized (pd.Series): HuggingFace BERT tokenized input.
    """
    
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(bert_method)
    
    # Loop through each item
    vector_tokenized = vector.copy()
    for i in range(0, len(vector)):
        vector_tokenized[i] = ' '.join(tokenizer.tokenize(vector[i]))
    
    return vector_tokenized

def recommend_with_tfidf(df_clean, text_col='cleaned_text', id_col='cord_uid', title_col='title', tokenization_method='scibert'):
    """ Recommend n items similar to the item of interest.
    
    Args:
        df_clean (pd.DataFrame): Cleaned dataframe.
        text_col (str): Column with cleaned text content.
        id_col (str): Column with ID.
        title_col (str): Column with item titles.
        tokenization_method (str): Either 'none', 'nltk', 'bert', or 'scibert'.
    
    Returns:
        results (dict): Dictionary containing ranked recommendations for all items.
    """
    token_dict = {}
    stemmer = PorterStemmer()

    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems

    # TF-IDF
    if tokenization_method.lower() in ['no', 'none']:
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')
        vector = df_clean[text_col]
    elif tokenization_method.lower() in ['nltk','stem','stemming']:
        tf = TfidfVectorizer(tokenizer=tokenize, analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')
        vector = df_clean[text_col]
    elif tokenization_method.lower() in ['huggingface','hf','bert']:
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')
        vector = tokenize_with_BERT(df_clean[text_col],'bert-base-cased')
    elif tokenization_method.lower() in ['scibert','sci-bert','sci_bert']:
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')
        vector = tokenize_with_BERT(df_clean[text_col], 'allenai/scibert_scivocab_cased')

    tfidf_matrix = tf.fit_transform(vector)

    # Similarity measure
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    results = {}
    for idx, row in df_clean.iterrows():
        similar_indices = cosine_sim[idx].argsort()[:-(len(df_clean)+1):-1]
        similar_items = [(cosine_sim[idx][i], df_clean[id_col][i], df_clean[title_col][i]) for i in similar_indices]
        results[row[id_col]] = similar_items[1:]
    
    return results

def organize_results_as_tabular(df_clean, results, id_col='cord_uid', k=5):
    """ Restructures results dictionary into a table.
    
    Args:
        df_clean (pd.DataFrame): Cleaned dataframe.
        results (dict): Dictionary containing ranked recommendations for all items.
        id_col (str): Column with ID.
        k (int): Number of wanted top recommendations.
    
    
    Returns:
        df_output (pd.DataFrame): Results organized into a tabular dataframe.
    """
    
    # Initialize new dataframe to hold recommendation output
    cord_uid = list()
    title = list()

    rec_rank = list()
    rec_score = list()

    rec_cord_uid = list()
    rec_title = list()

    for idx in range(0, len(results)):
        # Information about the item we are basing recommendations off of
        rec_based_on = list(results.keys())[idx]
        tmp_cord_uid = str(df_clean.loc[df_clean[id_col] == rec_based_on][id_col].values[0])
        tmp_title = str(df_clean.loc[df_clean[id_col] == rec_based_on]['title'].values[0])

        # Iterate through top k recommendations
        for i in range(0, k):
            # Save to lists
            cord_uid.append(tmp_cord_uid)
            title.append(tmp_title)
            rec_rank.append(str(i+1))
            rec_score.append(results[rec_based_on][i][0])
            rec_cord_uid.append(results[rec_based_on][i][1])
            rec_title.append(results[rec_based_on][i][2])

    
    # Save the output
    output_dict = {'cord_uid': cord_uid,
                   'title': title,
                   'rec_rank': rec_rank,
                   'rec_score': rec_score,
                   'rec_cord_uid': rec_cord_uid,
                   'rec_title': rec_title}

    # Convert to dataframe
    df_output = pd.DataFrame(output_dict)

    return df_output

def get_single_item_info(metadata, rec_id, id_col='cord_uid', verbose=True):
    """ Get full information for a single recommended item.
    
    Args:
        metadata (pd.DataFrame): Dataframe containing item info.
        rec_id (str): Identifier for recommended item.
        id_col (str): Column with IDs.
        verbose (boolean): Set to True if you want to print basic info and URL.
        
    Results:
        rec_info (pd.Series): Single row from dataframe containing recommended item info.
    """
    
    # Return row
    # rec_info = metadata.loc[metadata[id_col]==rec_id] # returns truncated info
    rec_info = metadata.iloc[int(np.where(metadata[id_col]==rec_id)[0])]

    if verbose == True:
        print('"'+ rec_info['title'] + '" (' + rec_info['publish_time'].rsplit('-')[0] + ')' + ' by ' + rec_info['authors'] + '.')
        print('Available at ' + rec_info['url'])
    
    return rec_info

def make_clickable(address):
    """ Make URL clickable.

    Args:
        address (str): URL address to make clickable.
    """
    return '<a href="{0}">{0}</a>'.format(address)

def display_top_recommendations(rec_table, metadata, query_id, id_col='cord_uid', verbose=True):
    """ Fill out the recommendation table with info about the recommendations.

    Args:
        rec_table (pd.DataFrame): Dataframe holding all recommendations.
        metadata (pd.DataFrame): Dataframe holding metadata for all public domain papers.
        query_id (str): ID of item of interest.
        id_col (str): Column with IDs.
        verbose (boolean): Set to True if you want to print the table.
    
    Results:
        df (pd.Styler): Stylized dataframe holding recommendations and associated metadata just for the item of interest (can access as normal dataframe by using df.data).
    """

    # Create subset of dataframe with just item of interest
    df = rec_table.loc[rec_table[id_col]==query_id].reset_index()

    # Remove id_col and title of query item
    df.drop([id_col, 'title'], axis=1, inplace=True)

    # Initialize new columns
    df['authors'] = np.nan
    df['journal'] = np.nan
    df['publish_time'] = np.nan
    df['url'] = np.nan

    # Add useful metadata
    for i in range(0,len(df)):
        rec_id = df['rec_'+id_col][i]
        rec_info = get_single_item_info(metadata, rec_id, id_col='cord_uid', verbose=False)
        df['authors'][i] = rec_info['authors']
        df['journal'][i] = rec_info['journal']
        df['publish_time'][i] = rec_info['publish_time']
        df['url'][i] = rec_info['url']

    # Rename columns such that rec_ is no longer appended, for simplicity
    df = df.rename(columns={'rec_rank': 'rank',
                            'rec_score': 'similarity_score',
                            'rec_title': 'title'})

    # Only keep columns of interest
    df = df[['rank', 'similarity_score', 'title', 'authors', 'journal', 'publish_time','url']]

    # Make URL clickable
    format_ = {'url': make_clickable}
    df = df.head().style.format(format_)

    if verbose == True:
        df

    return df