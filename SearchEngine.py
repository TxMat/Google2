import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from Corpus import Corpus

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (numpy.ndarray): The first vector.
        vec2 (numpy.ndarray): The second vector.

    Returns:
        float: The cosine similarity between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

class SearchEngine:
    """
    A class to represent a search engine.

    Attributes:
        term_freq_matrix (csr_matrix): The term frequency matrix of the corpus.
        vocab (list): The vocabulary of the corpus.
        corpus (Corpus): The corpus used by the search engine.
        doc_vectors (numpy.ndarray): The TF-IDF vectors of the documents in the corpus.
    """

    def __init__(self, corpus: Corpus):
        """
        Initialize the search engine with a given corpus.

        Args:
            corpus (Corpus): The corpus to use for the search engine.
        """
        self.term_freq_matrix = corpus.get_tf_matrix()
        self.vocab = corpus.get_vocab()
        self.corpus = corpus
        self.doc_vectors = self.calculate_tfidf_matrix().toarray()

    def calculate_tfidf_matrix(self):
        """
        Calculate the TF-IDF matrix for the corpus.

        Returns:
            csr_matrix: The TF-IDF matrix.
        """
        doc_freq = np.bincount(self.term_freq_matrix.indices, minlength=self.term_freq_matrix.shape[1])
        idf = np.log((1 + self.term_freq_matrix.shape[0]) / (1 + doc_freq)) + 1
        return self.term_freq_matrix.multiply(idf)

    def get_vector(self, query):
        """
        Convert a query into a vector based on the corpus vocabulary.

        Args:
            query (str): The search query.

        Returns:
            numpy.ndarray: The query vector.
        """
        cleaned_query = self.corpus.clean_text(query)
        query_vector = np.zeros(len(self.vocab))
        for term in cleaned_query.split():
            if term in self.vocab:
                query_vector[self.vocab.index(term)] += 1
        return query_vector

    def basic_search(self, query, source_list=None):
        """
        Perform a basic search on the corpus using cosine similarity.

        Args:
            query (str): The search query.
            source_list (list, optional): List of sources to filter the search results.

        Returns:
            DataFrame: The search results.
        """
        query_vector = self.get_vector(query)
        similarity = self.term_freq_matrix.dot(query_vector)
        results = []
        for i, score in tqdm(enumerate(similarity), ascii=True, desc="Searching (Basic)"):
            if source_list and self.corpus.id2doc[i + 1].source not in source_list:
                continue
            if score > 0:
                doc = self.corpus.id2doc[i+1]
                results.append([doc.body, score, doc.title, doc.author.name, doc.date, doc.url, doc.get_data()])
        results.sort(key=lambda x: x[1], reverse=True)
        return DataFrame(results, columns=["Body", "Score", "Title", "Author", "Date", "URL", "Document"])

    def advanced_search(self, query, source_list=None):
        """
        Perform an advanced search on the corpus using TF-IDF and cosine similarity.

        Args:
            query (str): The search query.
            source_list (list, optional): List of sources to filter the search results.

        Returns:
            DataFrame: The search results.
        """
        query_vector = self.get_vector(query)
        doc_freq = np.bincount(self.term_freq_matrix.indices, minlength=self.term_freq_matrix.shape[1])
        idf = np.log((1 + self.term_freq_matrix.shape[0]) / (1 + doc_freq)) + 1
        query_vector *= idf
        query_vector_norm = np.linalg.norm(query_vector)
        results = []
        for i in tqdm(range(self.doc_vectors.shape[0]), ascii=True, desc="Searching (Advanced)"):
            if source_list and self.corpus.id2doc[i + 1].source not in source_list:
                continue
            doc_vector = self.doc_vectors[i]
            similarity = np.dot(query_vector, doc_vector) / (query_vector_norm * np.linalg.norm(doc_vector))
            if similarity > 0:
                doc = self.corpus.id2doc[i+1]
                results.append([doc.body, similarity, doc.title, doc.author.name, doc.date, doc.url, doc.get_data()])
        results.sort(key=lambda x: x[1], reverse=True)
        return DataFrame(results, columns=["Body", "Score", "Title", "Author", "Date", "URL", "Document"])

    def bm25_search(self, query, k=1.5, b=0.65, source_list=None):
        """
        Perform a search on the corpus using the BM25 algorithm.

        Args:
            query (str): The search query.
            k (float, optional): The k parameter for BM25. Default is 1.5.
            b (float, optional): The b parameter for BM25. Default is 0.65.
            source_list (list, optional): List of sources to filter the search results.

        Returns:
            DataFrame: The search results.
        """
        query_vector = self.get_vector(query)
        idf = np.log((1 + self.term_freq_matrix.shape[0]) / (1 + np.bincount(self.term_freq_matrix.indices))) + 1
        avg_doc_length = np.mean([len(doc.body.split()) for doc in self.corpus.id2doc.values()])
        results = []
        for i in tqdm(range(self.doc_vectors.shape[0]), ascii=True, desc="Searching (BM25)"):
            if source_list and self.corpus.id2doc[i + 1].source not in source_list:
                continue
            doc_vector = self.doc_vectors[i]
            doc_length = len(self.corpus.id2doc[i + 1].body.split())
            similarity = bm25_score(query_vector, doc_vector, idf, doc_length, avg_doc_length, k, b)
            if similarity > 0:
                doc = self.corpus.id2doc[i + 1]
                results.append([doc.body, similarity, doc.title, doc.author.name, doc.date, doc.url, doc.get_data()])
        results.sort(key=lambda x: x[1], reverse=True)
        return DataFrame(results, columns=["Body", "Score", "Title", "Author", "Date", "URL", "Document"])

    def get_distinct_sources_list(self):
        """
        Get a list of distinct sources from the corpus.

        Returns:
            list: A list of distinct sources.
        """
        return self.corpus.get_distinct_sources_list()

def bm25_score(query_vector, doc_vector, idf, doc_length, avg_doc_length, k, b):
    """
    Calculate the BM25 score for a document.

    Args:
        query_vector (numpy.ndarray): The query vector.
        doc_vector (numpy.ndarray): The document vector.
        idf (numpy.ndarray): The inverse document frequency vector.
        doc_length (int): The length of the document.
        avg_doc_length (float): The average document length in the corpus.
        k (float): The k parameter for BM25.
        b (float): The b parameter for BM25.

    Returns:
        float: The BM25 score.
    """
    tf = doc_vector / (doc_vector + k * ((1 - b) + b * (doc_length / avg_doc_length)))
    return np.sum(tf * idf * query_vector)