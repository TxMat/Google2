import re

from pandas import DataFrame
from scipy.sparse import csr_matrix

from Author import Author
from Document import Document


class Corpus:
    nom: str
    authors: dict[int, Author]
    id2doc: dict[int, Document]
    cached_doc_string_list: str
    ndoc: int
    naut: int

    def __init__(self, nom):
        self.nom = nom
        self.authors = {}
        self.aut2id = {}  # todo id to author instead
        self.id2doc = {}
        self.cached_doc_string_list = ""
        self.ndoc = 0
        self.naut = 0

    doc: Document

    def add(self, doc: Document):
        if doc.author not in self.aut2id:
            self.naut += 1
            self.authors[self.naut] = Author(doc.author)
            self.aut2id[doc.author] = self.naut
        self.authors[self.aut2id[doc.author]].add_document(doc)

        self.ndoc += 1
        self.id2doc[self.ndoc] = doc
        self.cached_doc_string_list = ""

    def refresh_cache(self):
        self.cached_doc_string_list = "\n".join([doc.get_data() for doc in self.id2doc.values()])

    def search_regex(self, query: str):
        if not self.cached_doc_string_list:
            self.refresh_cache()
        return re.findall(query, self.cached_doc_string_list)

    def concordancer(self, query):
        if not self.cached_doc_string_list:
            self.refresh_cache()

        result = []

        for match in re.finditer(query, self.cached_doc_string_list):
            start = match.start()
            end = match.end()
            left_context = self.cached_doc_string_list[max(0, start - 30):start]
            right_context = self.cached_doc_string_list[end:min(len(self.cached_doc_string_list), end + 30)]
            result.append((left_context, self.cached_doc_string_list[start:end], right_context))

        return DataFrame(result, columns=["left context", "word", "right context"])

    @staticmethod
    def clean_text(text: str):
        text = text.lower()
        text = text.replace("\n", " ")
        text = re.sub(r"[^a-zà-ÿ@]", " ", text)
        return text

    def stats(self):
        if self.cached_doc_string_list == "":
            self.refresh_cache()

        vocab = set()
        freq = {}
        docu_freq = {}

        for doc in self.id2doc.values():
            doc_text = self.clean_text(doc.get_data())
            words = doc_text.split()
            unique_words = set(words)
            vocab.update(unique_words)

            for word in unique_words:
                if word not in docu_freq:
                    docu_freq[word] = 0
                docu_freq[word] += 1

            for word in words:
                if word not in freq:
                    freq[word] = 0
                freq[word] += 1

        freq_df = DataFrame(freq.items(), columns=["word", "frequency"])
        freq_df = freq_df.sort_values(by="frequency", ascending=False)

        docu_freq_df = DataFrame(docu_freq.items(), columns=["word", "document frequency"])
        freq_df = freq_df.merge(docu_freq_df, on="word")

        return freq_df

    def get_distinct_sources_list(self):
        sources = set()
        for doc in self.id2doc.values():
            sources.add(doc.source)
        return list(sources)

    def get_vocab(self):
        if self.cached_doc_string_list == "":
            self.refresh_cache()

        vocab = set()

        for doc in self.id2doc.values():
            doc_text = self.clean_text(doc.get_data())
            words = doc_text.split()
            unique_words = set(words)
            vocab.update(unique_words)

        vocab = list(vocab)
        vocab.sort()
        return vocab

    def get_tf_matrix(self):
        vocab = self.get_vocab()
        vocab2id = {word: i for i, word in enumerate(vocab)}

        rows = []
        cols = []
        data = []

        for i, doc in self.id2doc.items():
            doc_text = self.clean_text(doc.get_data())
            words = doc_text.split()
            for word in words:
                rows.append(i - 1)  # Adjusting index to be zero-based
                cols.append(vocab2id[word])
                data.append(1)

        tf_matrix = csr_matrix((data, (rows, cols)), shape=(len(self.id2doc), len(vocab)))

        return tf_matrix

    def show(self, n_docs=-1, tri="abc"):
        docs = list(self.id2doc.values())
        if tri == "abc":  # Tri alphabétique
            docs = list(sorted(docs, key=lambda x: x.titre.lower()))[:n_docs]
        elif tri == "123":  # Tri temporel
            docs = list(sorted(docs, key=lambda x: x.date))[:n_docs]

        print("\n".join(list(map(repr, docs))))

    def __repr__(self):
        docs = list(self.id2doc.values())
        docs = list(sorted(docs, key=lambda x: x.titre.lower()))

        return "\n".join(list(map(str, docs)))
