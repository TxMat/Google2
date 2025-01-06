import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from Corpus import Corpus


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)


class SearchEngine:

    def __init__(self, corpus : Corpus):
        self.term_freq_matrix = corpus.get_tf_matrix()
        self.vocab = corpus.get_vocab()
        self.corpus = corpus
        self.doc_vectors = self.calculate_tfidf_matrix().toarray()  # Precompute document vectors

    def calculate_tfidf_matrix(self):
        # Calculate document frequency for each term
        doc_freq = np.bincount(self.term_freq_matrix.indices, minlength=self.term_freq_matrix.shape[1])
        # Calculate inverse document frequency
        idf = np.log((1 + self.term_freq_matrix.shape[0]) / (1 + doc_freq)) + 1
        # Calculate TF-IDF matrix
        tfidf_matrix = self.term_freq_matrix.multiply(idf)
        return tfidf_matrix

    def get_vector(self, query):
        # transform query into vector
        cleaned_query = self.corpus.clean_text(query)
        query_vector = [0] * len(self.vocab)
        for term in cleaned_query.split():
            if term in self.vocab:
                query_vector[self.vocab.index(term)] += 1
        return np.array(query_vector)

    def basic_search(self, query, source_list=None):
        if source_list is None:
            source_list = []
        # transform query into vector
        query_vector = self.get_vector(query)
        # calculate similarity between query vector and all documents
        similarity = self.term_freq_matrix.dot(query_vector)
        # sort results and display the best results
        results = []
        for i, score in tqdm(enumerate(similarity), ascii=True, desc="Searching"):
            if source_list and self.corpus.id2doc[i + 1].source not in source_list:
                continue
            if score > 0:
                doc = self.corpus.id2doc[i+1]
                results.append([doc.body, score, doc.title, doc.author.name, doc.date, doc.url, doc.get_data()])
        results.sort(key=lambda x: x[1], reverse=True)
        return DataFrame(results, columns=["Body", "Score", "Title", "Author", "Date", "URL", "Document"])

    def advanced_search(self, query, source_list=None):
        if source_list is None:
            source_list = []
        # transform query into vector
        query_vector = self.get_vector(query)

        # Calculate document frequency for the query terms
        doc_freq = np.bincount(self.term_freq_matrix.indices, minlength=self.term_freq_matrix.shape[1])
        idf = np.log((1 + self.term_freq_matrix.shape[0]) / (1 + doc_freq)) + 1

        # Adjust the query vector using the inverse document frequency
        query_vector = query_vector * idf

        # calculate similarity between query vector and all documents
        query_vector_norm = np.linalg.norm(query_vector)
        results = []
        for i in tqdm(range(self.doc_vectors.shape[0]), ascii=True, desc="Searching (better)"):
            if source_list and self.corpus.id2doc[i + 1].source not in source_list:
                continue
            doc_vector = self.doc_vectors[i]
            similarity = np.dot(query_vector, doc_vector) / (query_vector_norm * np.linalg.norm(doc_vector))
            if similarity > 0:
                doc = self.corpus.id2doc[i+1]
                results.append([doc.body, similarity, doc.title, doc.author.name, doc.date, doc.url, doc.get_data()])

        # sort results and display the best results
        results.sort(key=lambda x: x[1], reverse=True)
        return DataFrame(results, columns=["Body", "Score", "Title", "Author", "Date", "URL", "Document"])


    def bm25_search(self, query, k=1.5, b=0.65, source_list=None):
        if source_list is None:
            source_list = []
        query_vector = self.get_vector(query)
        idf = np.log((1 + self.term_freq_matrix.shape[0]) / (1 + np.bincount(self.term_freq_matrix.indices))) + 1
        avg_doc_length = np.mean([len(doc.body.split()) for doc in self.corpus.id2doc.values()])
        results = []

        for i in tqdm(range(self.doc_vectors.shape[0]), ascii=True, desc="Searching (even better ??)"):
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
        return self.corpus.get_distinct_sources_list()


def bm25_score(query_vector, doc_vector, idf, doc_length, avg_doc_length, k, b):
    # BM25 formula
    tf = doc_vector / (doc_vector + k * ((1 - b) + b * (doc_length / avg_doc_length)))
    return np.sum(tf * idf * query_vector)