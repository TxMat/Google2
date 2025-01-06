import logging
import time
from typing import Dict

import coloredlogs
import pandas as pd
import praw
import urllib3
import xmltodict

from Author import Author
from Corpus import Corpus
from Document import Document
from SearchEngine import SearchEngine

# Dictionary to store documents with their IDs
id2doc = {}
# Dictionary to store authors with their IDs
id2aut: Dict[str, Author] = {}

# Logger setup
logger = logging.getLogger(__name__)

def reddit_import(subject: str, nb_doc: int):
    """
    Imports documents from a specified subreddit.

    Args:
        subject (str): The subreddit to import from.
        nb_doc (int): The number of documents to import.
    """
    reddit_client = praw.Reddit(client_id='T9qPLZ2H3esl_ukXzn0UVA', client_secret='zyGz9sEs67D0LB3Ihu3kia_3oq0KlQ', user_agent='search_engine_td')
    for key, submission in enumerate(reddit_client.subreddit(subject).hot(limit=nb_doc)):
        author_name = submission.author.name if submission.author else "Unknown"
        if author_name not in id2aut:
            id2aut[author_name] = Author(author_name)
        date = pd.to_datetime(int(submission.created_utc), utc=True, unit='s')
        doc = Document(submission.title, id2aut[author_name], date, submission.url, submission.selftext, "reddit")
        id2doc[f"red{key}"] = doc
        id2aut[author_name].add_document(doc)

def arxiv_import(subject: str, nb_doc: int):
    """
    Imports documents from arXiv based on a search query.

    Args:
        subject (str): The search query for arXiv.
        nb_doc (int): The number of documents to import.
    """
    resp = urllib3.request('GET', f'http://export.arxiv.org/api/query?search_query=all:{subject}&max_results={nb_doc}')
    articles = xmltodict.parse(resp.data)['feed']['entry']
    for key, article in enumerate(articles):
        title = article['title'].replace("\n", " ")
        summary = article['summary'].replace("\n", " ")
        author_name = article['author'][0]['name'] if isinstance(article['author'], list) else article['author']['name']
        if author_name not in id2aut:
            id2aut[author_name] = Author(author_name)
        date = pd.to_datetime(article['published'])
        doc = Document(title, id2aut[author_name], date, article['id'], summary, "arxiv")
        id2doc[f"arx{key}"] = doc
        id2aut[author_name].add_document(doc)

def us_speeches_import():
    """
    Imports US speeches from a CSV file.
    """
    us_data = pd.read_csv("data/discours_US.csv", sep="\t")

    counter = 0
    trump_author = Author("TRUMP")
    clinton_author = Author("CLINTON")
    id2aut["TRUMP"] = trump_author
    id2aut["CLINTON"] = clinton_author
    for i, row in us_data.iterrows():
        for sentence in row['text'].split("."):
            if row['speaker'] == "TRUMP":
                doc = Document(f"Sentence-{counter}", trump_author, row['date'], row['link'], sentence, "us")
                id2aut["TRUMP"].add_document(doc)
            else:
                doc = Document(f"Sentence-{counter}", clinton_author, row['date'], row['link'], sentence, "us")
                id2aut["CLINTON"].add_document(doc)
            id2doc[f"us{counter}"] = doc
            counter += 1

def build_corpus(subject: str, nb: int) -> Corpus:
    """
    Builds a corpus by importing documents from various sources.

    Args:
        subject (str): The subject to search for.
        nb (int): The number of documents to import from each source.

    Returns:
        Corpus: The built corpus containing all imported documents.
    """
    corpus = Corpus("Main Corpus")
    try:
        reddit_import(subject, nb)
    except Exception as e:
        logger.error(f"Error while importing reddit data: {e}, check that the subreddit exists")
    try:
        arxiv_import(subject, nb)
    except Exception as e:
        logger.error(f"Error while importing arxiv data: {e}")
    us_speeches_import()
    for doc in id2doc.values():
        corpus.add(doc)
    return corpus

def get_search_engine(corpus: Corpus) -> SearchEngine:
    """
    Initializes a search engine with the given corpus.

    Args:
        corpus (Corpus): The corpus to use for the search engine.

    Returns:
        SearchEngine: The initialized search engine.
    """
    return SearchEngine(corpus)

def init(subject: str, nb: int, should_build_corpus: bool) -> SearchEngine:
    """
    Initializes the search engine, either by building a new corpus or loading an existing one.

    Args:
        subject (str): The subject to search for.
        nb (int): The number of documents to import from each source.
        should_build_corpus (bool): Whether to build a new corpus or load an existing one.

    Returns:
        SearchEngine: The initialized search engine.
    """
    coloredlogs.install(level='INFO')
    start_time = time.time()
    if should_build_corpus:
        logger.info("Building corpus")
        corpus = build_corpus(subject, nb)
        save_corpus(corpus)
    else:
        logger.warning("should_build_corpus is set to False, loading corpus from file")
        corpus = load_corpus()
    logger.info(f"Corpus built in {round(time.time() - start_time, 2)} seconds")
    logger.info("Corpus stats:")
    print(corpus.stats())
    return get_search_engine(corpus)

def save_corpus(corpus: Corpus):
    """
    Saves the corpus to a file.

    Args:
        corpus (Corpus): The corpus to save.
    """
    import pickle
    with open("corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)

def load_corpus() -> Corpus:
    """
    Loads the corpus from a file.

    Returns:
        Corpus: The loaded corpus.
    """
    import pickle
    with open("corpus.pkl", "rb") as f:
        return pickle.load(f)

if __name__ == '__main__':
    init("usa", 100, False)