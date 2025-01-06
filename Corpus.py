import re
from pandas import DataFrame
from scipy.sparse import csr_matrix
from Author import Author
from Document import Document

class Corpus:
    """
    A class to represent a collection of documents (corpus).

    Attributes:
        nom (str): The name of the corpus.
        Authors (dict): A dictionary mapping author IDs to Author objects.
        Aut2id (dict): A dictionary mapping author names to author IDs.
        Id2doc (dict): A dictionary mapping document IDs to Document objects.
        Cached_doc_string_list (str): A cached string of all document data.
        Ndoc (int): The number of documents in the corpus.
        Naut (int): The number of authors in the corpus.
    """

    def __init__(self, nom):
        """
        Constructs all the necessary attributes for the Corpus object.

        Args:
            nom (str): The name of the corpus.
        """
        self.nom = nom
        self.authors = {}
        self.aut2id = {}
        self.id2doc = {}
        self.cached_doc_string_list = ""
        self.ndoc = 0
        self.naut = 0

    def add(self, doc: Document):
        """
        Adds a document to the corpus.

        Args:
            doc (Document): The document to add.
        """
        if doc.author not in self.aut2id:
            self.naut += 1
            self.authors[self.naut] = Author(doc.author)
            self.aut2id[doc.author] = self.naut
        self.authors[self.aut2id[doc.author]].add_document(doc)
        self.ndoc += 1
        self.id2doc[self.ndoc] = doc
        self.cached_doc_string_list = ""

    def refresh_cache(self):
        """
        Refreshes the cached string of all document data.
        """
        self.cached_doc_string_list = "\n".join(doc.get_data() for doc in self.id2doc.values())

    def search_regex(self, query: str):
        """
        Searches the corpus for a regex pattern.

        Args:
            query (str): The regex pattern to search for.

        Returns:
            list: A list of matches.
        """
        if not self.cached_doc_string_list:
            self.refresh_cache()
        return re.findall(query, self.cached_doc_string_list)

    def concordancer(self, query):
        """
        Creates a concordance for a regex pattern.

        Args:
            query (str): The regex pattern to search for.

        Returns:
            DataFrame: A DataFrame containing the left context, matched word, and right context.
        """
        if not self.cached_doc_string_list:
            self.refresh_cache()
        result = []
        for match in re.finditer(query, self.cached_doc_string_list):
            start, end = match.start(), match.end()
            left_context = self.cached_doc_string_list[max(0, start - 30):start]
            right_context = self.cached_doc_string_list[end:end + 30]
            result.append((left_context, self.cached_doc_string_list[start:end], right_context))
        return DataFrame(result, columns=["left context", "word", "right context"])

    @staticmethod
    def clean_text(text: str):
        """
        Cleans the text by converting to lowercase and removing non-alphabetic characters.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        text = text.lower().replace("\n", " ")
        return re.sub(r"[^a-zà-ÿ@]", " ", text)

    def stats(self):
        """
        Computes statistics for the corpus, including word frequency and document frequency.

        Returns:
            DataFrame: A DataFrame containing word frequency and document frequency.
        """
        if not self.cached_doc_string_list:
            self.refresh_cache()
        vocab, freq, docu_freq = set(), {}, {}
        for doc in self.id2doc.values():
            words = self.clean_text(doc.get_data()).split()
            unique_words = set(words)
            vocab.update(unique_words)
            for word in unique_words:
                docu_freq[word] = docu_freq.get(word, 0) + 1
            for word in words:
                freq[word] = freq.get(word, 0) + 1
        freq_df = DataFrame(freq.items(), columns=["word", "frequency"]).sort_values(by="frequency", ascending=False)
        docu_freq_df = DataFrame(docu_freq.items(), columns=["word", "document frequency"])
        return freq_df.merge(docu_freq_df, on="word")

    def get_distinct_sources_list(self):
        """
        Gets a list of distinct sources from the corpus.

        Returns:
            list: A list of distinct sources.
        """
        return list({doc.source for doc in self.id2doc.values()})

    def get_vocab(self):
        """
        Gets the vocabulary of the corpus.

        Returns:
            list: A sorted list of unique words in the corpus.
        """
        if not self.cached_doc_string_list:
            self.refresh_cache()
        vocab = {word for doc in self.id2doc.values() for word in self.clean_text(doc.get_data()).split()}
        return sorted(vocab)

    def get_tf_matrix(self):
        """
        Gets the term frequency matrix for the corpus.

        Returns:
            csr_matrix: The term frequency matrix.
        """
        vocab = self.get_vocab()
        vocab2id = {word: i for i, word in enumerate(vocab)}
        rows, cols, data = [], [], []
        for i, doc in self.id2doc.items():
            for word in self.clean_text(doc.get_data()).split():
                rows.append(i - 1)
                cols.append(vocab2id[word])
                data.append(1)
        return csr_matrix((data, (rows, cols)), shape=(len(self.id2doc), len(vocab)))

    def show(self, n_docs=-1, tri="abc"):
        """
        Displays the documents in the corpus.

        Args:
            n_docs (int, optional): The number of documents to display. Defaults to -1 (all documents).
            tri (str, optional): The sorting criterion ("abc" for alphabetical, otherwise by date). Defaults to "abc".
        """
        docs = sorted(self.id2doc.values(), key=lambda x: x.titre.lower() if tri == "abc" else x.date)[:n_docs]
        print("\n".join(map(repr, docs)))

    def __repr__(self):
        """
        Returns a string representation of the corpus.

        Returns:
            str: A string representation of the corpus.
        """
        return "\n".join(map(str, sorted(self.id2doc.values(), key=lambda x: x.titre.lower())))