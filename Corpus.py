import re
from pandas import DataFrame
from scipy.sparse import csr_matrix
from Author import Author
from Document import Document

class Corpus:
    def __init__(self, nom):
        self.nom = nom
        self.authors = {}
        self.aut2id = {}
        self.id2doc = {}
        self.cached_doc_string_list = ""
        self.ndoc = 0
        self.naut = 0

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
        self.cached_doc_string_list = "\n".join(doc.get_data() for doc in self.id2doc.values())

    def search_regex(self, query: str):
        if not self.cached_doc_string_list:
            self.refresh_cache()
        return re.findall(query, self.cached_doc_string_list)

    def concordancer(self, query):
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
        text = text.lower().replace("\n", " ")
        return re.sub(r"[^a-zà-ÿ@]", " ", text)

    def stats(self):
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
        return list({doc.source for doc in self.id2doc.values()})

    def get_vocab(self):
        if not self.cached_doc_string_list:
            self.refresh_cache()
        vocab = {word for doc in self.id2doc.values() for word in self.clean_text(doc.get_data()).split()}
        return sorted(vocab)

    def get_tf_matrix(self):
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
        docs = sorted(self.id2doc.values(), key=lambda x: x.titre.lower() if tri == "abc" else x.date)[:n_docs]
        print("\n".join(map(repr, docs)))

    def __repr__(self):
        return "\n".join(map(str, sorted(self.id2doc.values(), key=lambda x: x.titre.lower())))