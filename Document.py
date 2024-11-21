# split into redditdoc and arkivdoc ??
from datetime import datetime


class Document:

    title: str
    author: "Author" # todo should be a list
    date: datetime.date
    url: str
    body: str

    # todo source ?

    def __init__(self, title, author, date, url, body) -> None:
        self.title = title
        self.author = author
        self.date = date
        self.url = url
        self.body = body


    def get_data(self):
        return self.title + " " + self.author.name + " " + self.body

    def __str__(self) -> str:
        return f"{self.title} by {self.author.name}"

    def __repr__(self) -> str:
        return f"{self.title} by {self.author.name}"

    def pretty_print(self) -> None:
        print(f"Title: {self.title}")
        print(f"Author: {self.author}")
        print(f"Date: {self.date}")
        print(f"URL: {self.url}")
        print(f"Body: {self.body}")
