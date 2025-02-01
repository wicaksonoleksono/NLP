import pandas as pd
import random
from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from itertools import combinations, chain


class createPairs:
    def __init__(self, df, drop=True):

        self.df = df.copy()
        if drop:
            self.df = self._drop_non_english(self.df)
        self.stop_words = set(stopwords.words('english'))
    
    def _drop_non_english(self, df, content_column='content'):
        def is_english(text):
            try:
                return detect(text) == 'en'
            except:
                return False  
        return df[df[content_column].apply(is_english)].reset_index(drop=True)
    
    def _tokenize_text(self, text):
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
        tokens = word_tokenize(cleaned_text)
        filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in self.stop_words]
        return filtered_tokens
    
    def tokenize_all(self, column='content'):
        tokenized_texts = []
        for text in self.df[column].dropna():
            tokens = self._tokenize_text(text)
            if tokens:  # Ensure the list is not empty
                tokenized_texts.append(tokens)
        return tokenized_texts
    
    def build_vocab(self, tokenized_texts):
        vocab = set(chain.from_iterable(tokenized_texts))
        return sorted(vocab)
    
    def create_training_pairs(self, tokenized_texts):
        training_pairs = []
        for tokens in tokenized_texts:
            if len(tokens) < 2:
                continue  # Skip if less than two tokens
            # Create adjacent pairs
            for i in range(len(tokens) - 1):
                pair_forward = (tokens[i], tokens[i+1])
                pair_backward = (tokens[i+1], tokens[i])
                training_pairs.append(pair_forward)
                training_pairs.append(pair_backward)
        return training_pairs
    
    def get_vocab_and_pairs(self, column='content'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        tokenized_texts = self.tokenize_all(column)
        vocab = self.build_vocab(tokenized_texts)
        training_pairs = self.create_training_pairs(tokenized_texts)
        return vocab, training_pairs