import numpy as np
from pandas import DataFrame


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)


class SearchEngine:

    def __init__(self, corpus):
        self.term_freq_matrix = corpus.get_tf_matrix()
        self.vocab = corpus.get_vocab()
        self.TFxIDF_matrix = self.calculate_tfidf_matrix()
        self.corpus = corpus

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
        return query_vector

    def search(self, query):
        # transform query into vector
        query_vector = self.get_vector(query)
        # calculate similarity between query vector and all documents
        similarity = self.term_freq_matrix.dot(query_vector)
        # sort results and display the best results
        results = []
        for i, score in enumerate(similarity):
            if score > 0:
                doc = self.corpus.id2doc[i+1]
                results.append([doc.get_data(), score, doc.title, doc.author.name, doc.date, doc.url, doc.body])
        results.sort(key=lambda x: x[1], reverse=True)
        return DataFrame(results, columns=["Document", "Score", "Title", "Author", "Date", "URL", "Body"])

    def better_search(self, query):
        # transform query into vector
        query_vector = np.array(self.get_vector(query))

        # Calculate document frequency for the query terms
        doc_freq = np.bincount(self.term_freq_matrix.indices, minlength=self.term_freq_matrix.shape[1])
        idf = np.log((1 + self.term_freq_matrix.shape[0]) / (1 + doc_freq)) + 1

        # Adjust the query vector using the inverse document frequency
        query_vector = query_vector * idf

        # calculate similarity between query vector and all documents
        results = []
        for i in range(self.TFxIDF_matrix.shape[0]):
            doc_vector = self.TFxIDF_matrix.getrow(i).toarray().flatten()
            similarity = cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                doc = self.corpus.id2doc[i+1]
                results.append([doc.get_data(), similarity, doc.title, doc.author.name, doc.date, doc.url, doc.body])

        # sort results and display the best results
        results.sort(key=lambda x: x[1], reverse=True)
        return DataFrame(results, columns=["Document", "Score", "Title", "Author", "Date", "URL", "Body"])
