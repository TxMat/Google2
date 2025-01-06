from typing import List

from Document import Document


class Author:
    name: str
    ndoc: int
    production: List[Document]

    def __init__(self, name, ndoc=0, production=None):
        if production is None:
            production = []
        self.name = name
        self.ndoc = ndoc
        self.production = production

    def add_document(self, document: Document):
        self.production.append(document)
        self.ndoc += 1

    def __str__(self):
        return f"{self.name} - {self.ndoc} publications"

    def __repr__(self):
        return f"{self.name} - {self.ndoc} publications"

    def pretty_print(self):
        print(f"Name: {self.name}")
        print(f"Ndoc: {self.ndoc}")
        print(f"Production: {self.production}")

    def stats(self):
        print(f"Published {self.ndoc} publications")
        text_sum = 0
        for d in self.production:
            text_sum += len(d.body)
        print(f"average doc length: {text_sum / len(self.production)}")
