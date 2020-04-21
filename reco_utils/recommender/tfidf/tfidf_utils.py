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

class TfidfRecommender:
    """Term Frequency - Inverse Document Frequency (TF-IDF) Recommender

    This class provides content-based recommendations using TF-IDF vectorization in combination with cosine similarity.
    """

    def __init__(self, tokenization_method, id_col, title_col):
        """Initialize model parameters

        Args:
            tokenization_method (str): ['none','nltk','bert','scibert'] option for tokenization method.
            id_col (str): Name of column containing item IDs.
            title_col (str): Name of column containing item titles.
        """
        
        if tokenization_method.lower() not in ['none','nltk','bert','scibert']:
            raise ValueError(
                'Tokenization method must be one of ["none" | "nltk" | "bert" | "scibert"]'
            )
        self.tokenization_method = tokenization_method.lower()
        self.id_col = id_col
        self.title_col = title_col

        # Initialize other variables used in this class
        self.tf = TfidfVectorizer()
        self.tfidf_matrix = dict()
        self.tokens = dict()
        self.stop_words = frozenset()
        self.recommendations = dict()
        self.top_k_recommendations = pd.DataFrame()
        
    def __clean_text(self, text, for_BERT=False, verbose=False):
        """ Clean text by removing HTML tags, symbols, and punctuation.
        
        Args:
            text (str): Text to clean.
            for_BERT (boolean): True or False for if this text is being cleaned for a BERT word tokenization method.
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

    def clean_dataframe(self, df, cols_to_clean, new_col_name='cleaned_text'):
        """ Clean the text within the columns of interest and return a dataframe with cleaned and combined text.
        
        Args:
            df (pd.DataFrame): Dataframe containing the text content to clean.
            cols_to_clean (str): List of columns to clean by name (e.g., ['abstract','full_text']).
            new_col_name (str): Name of the new column that will contain the cleaned text.

        Returns:
            df (pd.DataFrame): Dataframe with cleaned text in the new column.
        """
        # Check if for BERT tokenization
        if self.tokenization_method in ['bert','scibert']:
            for_BERT = True
        else:
            for_BERT = False

        # Clean the text in the dataframe
        for i in range(0, len(df)):
            for col in cols_to_clean:
                df[col][i] = self.__clean_text(df[col][i], for_BERT)

        # Collapse the table such that all descriptive text is just in a single column
        first_col = True
        for col in cols_to_clean:
            if first_col is True:
                # first one
                df[new_col_name] = df[col]
                first_col = False
            else:
                df[new_col_name] = df[new_col_name] + ' ' + df[col]
        
        # Make sure any punctuation or special characters in Name are human readible
        for i in range(0, len(df)):
            df[self.title_col][i] = df[self.title_col][i].encode('ascii','ignore')
        
        for i in range(0, len(df)):
            df[self.title_col][i] = df[self.title_col][i].decode()
        
        return df
    
    def tokenize_text(self, df_clean, text_col='cleaned_text', ngram_range=(1,3), min_df=0):
        """ Tokenize the input text.
            For more details on the TfidfVectorizer, see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        
        Args:
            df_clean (pd.DataFrame): Dataframe with cleaned text in the new column.
            text_col (str): Name of column containing the cleaned text.
            ngram_range (tuple of int): The lower and upper boundary of the range of n-values for different n-grams to be extracted.
            min_df (int): When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
        
        Returns:
            tf (TfidfVectorizer): Scikit-learn TfidfVectorizer object defined in .tokenize_text().
            vectors_tokenized (pd.Series): Each row contains tokens for respective documents separated by spaces.
        """
        vectors = df_clean[text_col]

        # If a HuggingFace BERT word tokenization method
        if self.tokenization_method in ['bert','scibert']:
            # Set vectorizer
            tf = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=min_df, stop_words='english')

            # Get appropriate transformer name
            if self.tokenization_method == 'bert':
                bert_method = 'bert-base-cased'
            elif self.tokenization_method == 'scibert':
                bert_method = 'allenai/scibert_scivocab_cased'
            
            # Load pre-trained model tokenizer (vocabulary)
            tokenizer = BertTokenizer.from_pretrained(bert_method)
            
            # Loop through each item
            vectors_tokenized = vectors.copy()
            for i in range(0, len(vectors)):
                vectors_tokenized[i] = ' '.join(tokenizer.tokenize(vectors[i]))

        elif self.tokenization_method == 'nltk':
            # NLTK Stemming
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
            
            # When defining a custome tokenizer with TfidfVectorizer, the tokenization is applied in the fit function
            tf = TfidfVectorizer(tokenizer=tokenize, analyzer='word', ngram_range=ngram_range, min_df=min_df, stop_words='english')
            vectors_tokenized = vectors

        elif self.tokenization_method == 'none':
            # No tokenization applied
            tf = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=min_df, stop_words='english')
            vectors_tokenized = vectors
        
        # Save to class variable
        self.tf = tf

        return tf, vectors_tokenized
    
    def fit(self, tf, vectors_tokenized):
        """ Fit TF-IDF vectorizer to the cleaned and tokenized text.

        Args:
            tf (TfidfVectorizer): Scikit-learn TfidfVectorizer object defined in .tokenize_text().
            vectors_tokenized (pd.Series): Each row contains tokens for respective documents separated by spaces.
        """
        self.tfidf_matrix = tf.fit_transform(vectors_tokenized)

    def get_tokens(self):
        """ Return the tokens generated by the TF-IDF vectorizer.

        Returns:
            self.tokens (dict): Dictionary of tokens generated by the TF-IDF vectorizer.
        """
        try:
            self.tokens = self.tf.vocabulary_
        except:
            self.tokens = 'Run .tokenize_text() and .fit_tfidf() first'
        return self.tokens
    
    def get_stop_words(self):
        """ Return the stop words excluded in the TF-IDF vectorizer.

        Returns:
            self.stop_words (frozenset): Frozenset of stop words used by the TF-IDF vectorizer (can be converted to list).
        """
        try:
            self.stop_words = self.tf.get_stop_words()
        except:
            self.stop_words = 'Run .tokenize_text() and .fit_tfidf() first'
        return self.stop_words
    
    def __create_full_recommendation_dictionary(self, df_clean):
        """ Create the full recommendation dictionary containing all recommendations for all items.
        
        Args:
            df_clean (pd.DataFrame): Dataframe with cleaned text.
        """

        # Similarity measure
        cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        results = {}
        for idx, row in df_clean.iterrows():
            similar_indices = cosine_sim[idx].argsort()[:-(len(df_clean)+1):-1]
            similar_items = [(cosine_sim[idx][i], df_clean[self.id_col][i], df_clean[self.title_col][i]) for i in similar_indices]
            results[row[self.id_col]] = similar_items[1:]
        
        # Save to class
        self.recommendations = results
    
    def __organize_results_as_tabular(self, df_clean, k):
        """ Restructures results dictionary into a table containing only the top k recommendations per item.
        
        Args:
            df_clean (pd.DataFrame): Dataframe with cleaned text.
            k (int): Number of recommendations to return.
        """
        
        # Initialize new dataframe to hold recommendation output
        item_id = list()
        title = list()

        rec_rank = list()
        rec_score = list()

        rec_item_id = list()
        rec_title = list()

        for idx in range(0, len(self.recommendations)):
            # Information about the item we are basing recommendations off of
            rec_based_on = list(self.recommendations.keys())[idx]
            tmp_item_id = str(df_clean.loc[df_clean[self.id_col] == rec_based_on][self.id_col].values[0])
            tmp_title = str(df_clean.loc[df_clean[self.id_col] == rec_based_on][self.title_col].values[0])

            # Iterate through top k recommendations
            for i in range(0, k):
                # Save to lists
                item_id.append(tmp_item_id)
                title.append(tmp_title)
                rec_rank.append(str(i+1))
                rec_score.append(self.recommendations[rec_based_on][i][0])
                rec_item_id.append(self.recommendations[rec_based_on][i][1])
                rec_title.append(self.recommendations[rec_based_on][i][2])

        
        # Save the output
        output_dict = {self.id_col: item_id,
                    'title': title,
                    'rec_rank': rec_rank,
                    'rec_score': rec_score,
                    'rec_'+self.id_col: rec_item_id,
                    'rec_title': rec_title}

        # Convert to dataframe
        self.top_k_recommendations = pd.DataFrame(output_dict)
    
    def recommend_top_k_items(self, df_clean, k=5):
        """ Recommend k number of items similar to the item of interest.

        Args:
            df_clean (pd.DataFrame): Dataframe with cleaned text.
            k (int): Number of recommendations to return.
        
        Returns:
            self.top_k_recommendations (pd.DataFrame): Dataframe containing id and title of top k recommendations for all items.
        """
        self.__create_full_recommendation_dictionary(df_clean)
        self.__organize_results_as_tabular(df_clean, k)

        return self.top_k_recommendations
    
    def __get_single_item_info(self, metadata, rec_id, verbose=True):
        """ Get full information for a single recommended item.
        
        Args:
            metadata (pd.DataFrame): Dataframe containing item info.
            rec_id (str): Identifier for recommended item.
            verbose (boolean): Set to True if you want to print basic info and URL.
            
        Results:
            rec_info (pd.Series): Single row from dataframe containing recommended item info.
        """
        
        # Return row
        rec_info = metadata.iloc[int(np.where(metadata[self.id_col]==rec_id)[0])]

        if verbose == True:
            print('"'+ rec_info['title'] + '" (' + rec_info['publish_time'].rsplit('-')[0] + ')' + ' by ' + rec_info['authors'] + '.')
            print('Available at ' + rec_info['url'])
        
        return rec_info

    def __make_clickable(self, address):
        """ Make URL clickable.

        Args:
            address (str): URL address to make clickable.
        """
        return '<a href="{0}">{0}</a>'.format(address)
    
    def get_top_k_recommendations(self, metadata, query_id, verbose=True):
        """ Return the top k recommendations with useful metadata for each recommendation.

        Args:
            metadata (pd.DataFrame): Dataframe holding metadata for all public domain papers.
            query_id (str): ID of item of interest.
            verbose (boolean): Set to True if you want to print the table.
        
        Results:
            df (pd.Styler): Stylized dataframe holding recommendations and associated metadata just for the item of interest (can access as normal dataframe by using df.data).
        """

        # Create subset of dataframe with just item of interest
        df = self.top_k_recommendations.loc[self.top_k_recommendations[self.id_col]==query_id].reset_index()

        # Remove id_col and title of query item
        df.drop([self.id_col, 'title'], axis=1, inplace=True)

        # Initialize new columns
        df['authors'] = np.nan
        df['journal'] = np.nan
        df['publish_time'] = np.nan
        df['url'] = np.nan

        # Add useful metadata
        for i in range(0,len(df)):
            rec_id = df['rec_'+self.id_col][i]
            rec_info = self.__get_single_item_info(metadata, rec_id, verbose=False)
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
        format_ = {'url': self.__make_clickable}
        df = df.head().style.format(format_)

        if verbose == True:
            df

        return df