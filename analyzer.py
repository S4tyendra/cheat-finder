import argparse
import os
from pathlib import Path

# Import necessary libraries (will add more specific ones later)
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Jaccard calculation might need custom implementation or NLTK

# Ensure necessary NLTK data is downloaded (e.g., tokenizers, stopwords)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
# Add lemmatizer download if needed
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')


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
    # TODO: Implement stemming/lemmatization and stopword removal options
    tokens = nltk.word_tokenize(text.lower())
    # Placeholder for actual preprocessing
    processed_text = ' '.join(tokens) # Rejoin for vectorizers for now
    return processed_text

def calculate_cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    # Ensure vectors are 2D arrays for cosine_similarity function
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

def calculate_jaccard_similarity(text1, text2, n=1):
    """Calculates Jaccard similarity based on token sets or n-grams."""
    # TODO: Implement Jaccard for tokens (n=1) and n-grams
    tokens1 = set(nltk.word_tokenize(text1.lower()))
    tokens2 = set(nltk.word_tokenize(text2.lower()))
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0

def analyze_similarity(doc1_content, doc2_content, use_stemming=False, remove_stopwords=True):
    """Calculates and returns similarity scores using various methods."""
    results = {}

    # --- Preprocessing (apply consistently) ---
    # This part needs refinement based on how vectorizers/jaccard handle preprocessed text
    # For now, just pass raw text and let vectorizers handle tokenization/stopwords
    # processed_doc1 = preprocess_text(doc1_content, use_stemming, remove_stopwords)
    # processed_doc2 = preprocess_text(doc2_content, use_stemming, remove_stopwords)
    corpus = [doc1_content, doc2_content] # Vectorizers work on a corpus

    # --- CountVectorizer + Cosine Similarity ---
    try:
        count_vectorizer = CountVectorizer(stop_words='english' if remove_stopwords else None)
        count_matrix = count_vectorizer.fit_transform(corpus)
        results['Cosine (CountVec)'] = calculate_cosine_similarity(count_matrix[0], count_matrix[1])
    except Exception as e:
        results['Cosine (CountVec)'] = f"Error: {e}"

    # --- TF-IDF Vectorizer + Cosine Similarity ---
    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english' if remove_stopwords else None)
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        results['Cosine (TF-IDF)'] = calculate_cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    except Exception as e:
        results['Cosine (TF-IDF)'] = f"Error: {e}"

    # --- Jaccard Similarity (Tokens) ---
    try:
        # Apply preprocessing manually for Jaccard
        tokens1 = nltk.word_tokenize(doc1_content.lower())
        tokens2 = nltk.word_tokenize(doc2_content.lower())
        if remove_stopwords:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            tokens1 = [w for w in tokens1 if w.isalnum() and w not in stop_words]
            tokens2 = [w for w in tokens2 if w.isalnum() and w not in stop_words]
        else:
             tokens1 = [w for w in tokens1 if w.isalnum()]
             tokens2 = [w for w in tokens2 if w.isalnum()]
        # TODO: Add stemming option here if use_stemming is True
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        results['Jaccard (Tokens)'] = intersection / union if union > 0 else 0
    except Exception as e:
        results['Jaccard (Tokens)'] = f"Error: {e}"


    # --- Jaccard Similarity (N-grams, e.g., trigrams) ---
    n = 3 # Define n outside the try block
    try:
        # Apply preprocessing manually for Jaccard N-grams
        tokens1_ng = nltk.word_tokenize(doc1_content.lower())
        tokens2_ng = nltk.word_tokenize(doc2_content.lower())
        if remove_stopwords:
            stop_words = set(nltk.corpus.stopwords.words('english'))
            tokens1_ng = [w for w in tokens1_ng if w.isalnum() and w not in stop_words]
            tokens2_ng = [w for w in tokens2_ng if w.isalnum() and w not in stop_words]
        else:
             tokens1_ng = [w for w in tokens1_ng if w.isalnum()]
             tokens2_ng = [w for w in tokens2_ng if w.isalnum()]
        # TODO: Add stemming option here if use_stemming is True

        ngrams1 = set(nltk.ngrams(tokens1_ng, n))
        ngrams2 = set(nltk.ngrams(tokens2_ng, n))
        intersection_ng = len(ngrams1.intersection(ngrams2))
        union_ng = len(ngrams1.union(ngrams2))
        results[f'Jaccard ({n}-grams)'] = intersection_ng / union_ng if union_ng > 0 else 0
    except Exception as e:
        results[f'Jaccard ({n}-grams)'] = f"Error: {e}"


    return results

def main():
    parser = argparse.ArgumentParser(description="Document Similarity Analyzer")
    parser.add_argument("doc1", help="Path to the first document")
    parser.add_argument("doc2", help="Path to the second document")
    parser.add_argument("--stemming", action="store_true", help="Enable stemming (Porter)")
    parser.add_argument("--no-stopwords", dest="remove_stopwords", action="store_false", help="Disable stop word removal")
    parser.set_defaults(remove_stopwords=True)

    args = parser.parse_args()

    doc1_path = Path(args.doc1)
    doc2_path = Path(args.doc2)

    # Load documents
    doc1_content = load_document(doc1_path)
    doc2_content = load_document(doc2_path)

    if doc1_content is None or doc2_content is None:
        print("Exiting due to file loading errors.")
        return

    print(f"Comparing '{doc1_path.name}' and '{doc2_path.name}':")
    print(f"Options: Stemming={args.stemming}, Remove Stopwords={args.remove_stopwords}\n")

    # Analyze similarity
    similarity_scores = analyze_similarity(
        doc1_content,
        doc2_content,
        use_stemming=args.stemming,
        remove_stopwords=args.remove_stopwords
    )

    # Display results
    print("--- Similarity Scores ---")
    for method, score in similarity_scores.items():
        if isinstance(score, float):
            print(f"{method:<20}: {score:.4f}")
        else:
            print(f"{method:<20}: {score}") # Print errors directly

if __name__ == "__main__":
    main()