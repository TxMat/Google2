import unittest

import numpy as np

from Author import Author
from Corpus import Corpus
from Document import Document
from SearchEngine import SearchEngine, cosine_similarity, bm25_score


class TestSearchEngine(unittest.TestCase):

    def setUp(self):
        self.corpus = Corpus("Test Corpus")
        self.author = Author("Test Author")
        self.doc1 = Document("Title1", self.author, "2023-01-01", "http://example.com/1", "This is a test document.",
                             "source1")
        self.doc2 = Document("Title2", self.author, "2023-01-02", "http://example.com/2", "Another test document.",
                             "source2")
        self.corpus.add(self.doc1)
        self.corpus.add(self.doc2)
        self.search_engine = SearchEngine(self.corpus)

    def test_calculates_cosine_similarity_correctly(self):
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        similarity = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.9746318461970762)

    def test_performs_basic_search_correctly(self):
        results = self.search_engine.basic_search("test")
        self.assertEqual(len(results), 2)
        self.assertIn("This is a test document.", results["Body"].values)
        self.assertIn("Another test document.", results["Body"].values)

    def test_performs_advanced_search_correctly(self):
        results = self.search_engine.advanced_search("test")
        self.assertEqual(len(results), 2)
        self.assertIn("This is a test document.", results["Body"].values)
        self.assertIn("Another test document.", results["Body"].values)

    def test_performs_bm25_search_correctly(self):
        results = self.search_engine.bm25_search("test")
        self.assertEqual(len(results), 2)
        self.assertIn("This is a test document.", results["Body"].values)
        self.assertIn("Another test document.", results["Body"].values)

    def test_handles_empty_query_in_basic_search(self):
        results = self.search_engine.basic_search("")
        self.assertEqual(len(results), 0)

    def test_handles_empty_query_in_advanced_search(self):
        results = self.search_engine.advanced_search("")
        self.assertEqual(len(results), 0)

    def test_handles_empty_query_in_bm25_search(self):
        results = self.search_engine.bm25_search("")
        self.assertEqual(len(results), 0)

    def test_filters_results_by_source_in_basic_search(self):
        results = self.search_engine.basic_search("test", source_list=["source1"])
        self.assertEqual(len(results), 1)
        self.assertIn("This is a test document.", results["Body"].values)

    def test_filters_results_by_source_in_advanced_search(self):
        results = self.search_engine.advanced_search("test", source_list=["source1"])
        self.assertEqual(len(results), 1)
        self.assertIn("This is a test document.", results["Body"].values)

    def test_filters_results_by_source_in_bm25_search(self):
        results = self.search_engine.bm25_search("test", source_list=["source1"])
        self.assertEqual(len(results), 1)
        self.assertIn("This is a test document.", results["Body"].values)

    def test_calculates_bm25_score_correctly(self):
        query_vector = np.array([1, 0, 1])
        doc_vector = np.array([1, 1, 1])
        idf = np.array([1.5, 1.0, 1.5])
        score = bm25_score(query_vector, doc_vector, idf, 3, 2, 1.5, 0.75)
        self.assertAlmostEqual(score, 1, 1)


if __name__ == '__main__':
    unittest.main()
