from typing import Dict, Any, Tuple

import praw
import urllib3
import xmltodict
import pandas as pd
from praw.models import Submission

from Author import Author
from Corpus import Corpus
from Document import Document
from SearchEngine import SearchEngine

id2doc = {}
id2aut: dict[str, Author] = {}

def reddit():
    reddit_client = praw.Reddit(client_id='T9qPLZ2H3esl_ukXzn0UVA', client_secret='zyGz9sEs67D0LB3Ihu3kia_3oq0KlQ',
                                user_agent='search_engine_td')
    key = 0
    submission: Submission
    # todo sbr title should be a param
    for submission in reddit_client.subreddit("linux").hot(limit=100):
        if submission.author.name not in id2aut:
            id2aut[submission.author.name] = Author(submission.author.name)
        date = pd.to_datetime(int(submission.created_utc), utc=True, unit='s')
        doc: Document = Document(submission.title, id2aut[submission.author.name], date, submission.url, submission.selftext)
        id2doc[f"red{key}"] = doc
        id2aut[submission.author.name].add_document(doc)
        key+=1


def arxiv():
    resp = urllib3.request('GET', 'http://export.arxiv.org/api/query?search_query=all:electron&max_results=100')
    generated_dict = xmltodict.parse(resp.data)

    key = 0

    for articles in generated_dict['feed']['entry']:
        title = articles['title'].replace("\n", " ")
        txt = articles['summary'].replace("\n", " ")
        if len(articles['author']) == 1:
            a = articles['author']["name"]
        elif len(articles['author']) > 1:
            if type(articles['author']) is list:
                a = articles['author'][0]['name']
            else:
                a = "Cannot parse"
        else:
            a = "Unknown"

        if a not in id2aut:
            id2aut[a] = Author(a)

        date = pd.to_datetime(articles['published'])
        doc = Document(title, id2aut[a], date, articles['id'], txt)
        id2doc[f"arx{key}"] = doc
        id2aut[a].add_document(doc)
        key+=1



def main():
    # reddit()
    # arxiv()
    #
    # corpus = Corpus("corpus")

    corpus = load_corpus()

    d: Document
    for d in id2doc.values():
        corpus.add(d)

    # print(corpus.search_regex("electron*.+"))
    # print(corpus.concordancer("electron").head())
    # print(corpus.stats())
    save_corpus(corpus)
    google = SearchEngine(corpus)
    print(google.search("fedora linux"))
    a = google.better_search("fedora linux")
    print(a[["Title", "Score", "URL"]])


def save_corpus(corpus: Corpus):
    import pickle

    with open("corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)

def load_corpus() -> Corpus:
    import pickle

    with open("corpus.pkl", "rb") as f:
        return pickle.load(f)

if __name__ == '__main__':
    main()
