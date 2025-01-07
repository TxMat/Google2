import unittest

import numpy as np

from Author import Author
from Corpus import Corpus
from Document import Document
from SearchEngine import SearchEngine, cosine_similarity, bm25_score


class TestCorpus(unittest.TestCase):

    def setUp(self):
        self.corpus = Corpus("Test Corpus")
        self.author = Author("Test Author")
        self.doc1 = Document("Title1", self.author, "2023-01-01", "http://example.com/1", "This is a test document.",
                             "source1")
        self.doc2 = Document("Title2", self.author, "2023-01-02", "http://example.com/2", "Another test document.",
                             "source2")
        self.corpus.add(self.doc1)
        self.corpus.add(self.doc2)

    def test_adds_document_to_corpus(self):
        self.assertEqual(self.corpus.ndoc, 2)
        self.assertIn(self.doc1, self.corpus.id2doc.values())
        self.assertIn(self.doc2, self.corpus.id2doc.values())

    def test_refreshes_cache_correctly(self):
        self.corpus.refresh_cache()
        self.assertIn("This is a test document.", self.corpus.cached_doc_string_list)
        self.assertIn("Another test document.", self.corpus.cached_doc_string_list)

    def test_searches_regex_correctly(self):
        matches = self.corpus.search_regex(r"test")
        self.assertEqual(len(matches), 2)

    def test_cleans_text_correctly(self):
        cleaned_text = self.corpus.clean_text("This is a TEST document! 123")
        self.assertEqual("this is a test document", cleaned_text)

    def test_computes_stats_correctly(self):
        stats = self.corpus.stats()
        self.assertIn("test", stats["word"].values)
        self.assertIn("document", stats["word"].values)

    def test_gets_distinct_sources_list_correctly(self):
        sources = self.corpus.get_distinct_sources_list()
        self.assertIn("source1", sources)
        self.assertIn("source2", sources)

    def test_gets_vocab_correctly(self):
        vocab = self.corpus.get_vocab()
        self.assertIn("test", vocab)
        self.assertIn("document", vocab)

    def test_gets_tf_matrix_correctly(self):
        tf_matrix = self.corpus.get_tf_matrix()
        self.assertEqual(tf_matrix.shape, (2, len(self.corpus.get_vocab())))

    def test_shows_documents_correctly(self):
        self.corpus.show()
        self.assertIn("Title1", repr(self.corpus))
        self.assertIn("Title2", repr(self.corpus))

    def test_represents_corpus_correctly(self):
        repr_str = repr(self.corpus)
        self.assertIn("Title1", repr_str)
        self.assertIn("Title2", repr_str)


