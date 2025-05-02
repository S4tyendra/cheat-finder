import argparse
import os
from pathlib import Path

import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')


def load_document(filepath):
    """Loads text content from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def preprocess_text(text, use_stemming=False, remove_stopwords=True):
    """Basic preprocessing: tokenization, lowercasing, optionally stopwords and stemming."""
    tokens = nltk.word_tokenize(text.lower())
    if remove_stopwords:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    processed_text = ' '.join(tokens)
    return processed_text

def calculate_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    # Ensure vectors are 2D arrays for cosine_similarity function
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

def calculate_jaccard_similarity(text1, text2, n=1):
    """Calculates Jaccard similarity based on token sets or n-grams."""
    tokens1 = set(nltk.word_tokenize(text1.lower()))
    tokens2 = set(nltk.word_tokenize(text2.lower()))
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0

def analyze_similarity(doc1_content, doc2_content, use_stemming=False, remove_stopwords=True):
    """Calculates and returns similarity scores using various methods."""
    results = {}

    corpus = [doc1_content, doc2_content]

    # CountVectorizer + Cosine Similarity 
    try:
        count_vectorizer = CountVectorizer(stop_words='english' if remove_stopwords else None)
        count_matrix = count_vectorizer.fit_transform(corpus)
        results['Cosine (CountVec)'] = calculate_cosine_similarity(count_matrix[0], count_matrix[1])
    except Exception as e:
        results['Cosine (CountVec)'] = f"Error: {e}"

    #  TF-IDF Vectorizer + Cosine Similarity 
    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english' if remove_stopwords else None)
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        results['Cosine (TF-IDF)'] = calculate_cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    except Exception as e:
        results['Cosine (TF-IDF)'] = f"Error: {e}"

    #  Jaccard Similarity (Tokens) 
    try:
        # preprocessing manually for Jaccard
        tokens1 = nltk.word_tokenize(doc1_content.lower())
        tokens2 = nltk.word_tokenize(doc2_content.lower())
        if remove_stopwords:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            tokens1 = [w for w in tokens1 if w.isalnum() and w not in stop_words]
            tokens2 = [w for w in tokens2 if w.isalnum() and w not in stop_words]
        else:
             tokens1 = [w for w in tokens1 if w.isalnum()]
             tokens2 = [w for w in tokens2 if w.isalnum()]
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        results['Jaccard (Tokens)'] = intersection / union if union > 0 else 0
    except Exception as e:
        results['Jaccard (Tokens)'] = f"Error: {e}"


    #  Jaccard Similarity (N-grams, e.g., trigrams) 
    n = 3 
    try:
        # preprocessing manually for Jaccard N-grams
        tokens1_ng = nltk.word_tokenize(doc1_content.lower())
        tokens2_ng = nltk.word_tokenize(doc2_content.lower())
        if remove_stopwords:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            tokens1_ng = [w for w in tokens1_ng if w.isalnum() and w not in stop_words]
            tokens2_ng = [w for w in tokens2_ng if w.isalnum() and w not in stop_words]
        else:
             tokens1_ng = [w for w in tokens1_ng if w.isalnum()]
             tokens2_ng = [w for w in tokens2_ng if w.isalnum()]

        ngrams1 = set(nltk.ngrams(tokens1_ng, n))
        ngrams2 = set(nltk.ngrams(tokens2_ng, n))
        intersection_ng = len(ngrams1.intersection(ngrams2))
        union_ng = len(ngrams1.union(ngrams2))
        results[f'Jaccard ({n}-grams)'] = intersection_ng / union_ng if union_ng > 0 else 0
    except Exception as e:
        results[f'Jaccard ({n}-grams)'] = f"Error: {e}"


    return results
