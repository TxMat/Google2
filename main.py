import praw
import urllib3
import xmltodict
import pandas as pd
from praw.models import Submission

from Author import Author
from Document import Document

id2doc = {}
id2aut = {}

def reddit():
    reddit_client = praw.Reddit(client_id='T9qPLZ2H3esl_ukXzn0UVA', client_secret='zyGz9sEs67D0LB3Ihu3kia_3oq0KlQ',
                                user_agent='search_engine_td')
    key = 0
    submission: Submission
    # todo sbr title should be a param
    for submission in reddit_client.subreddit("france").hot(limit=10):
        if submission.author.name not in id2aut:
            id2aut[submission.author.name] = Author(submission.author.name)
        id2doc[f"red{key}"] = Document(submission.title, id2aut[submission.author.name], submission.created_utc, submission.url, submission.selftext)
        key+=1


def arxiv():
    resp = urllib3.request('GET', 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=10')
    generated_dict = xmltodict.parse(resp.data)

    key = 0

    for articles in generated_dict['feed']['entry']:
        title = articles['title'].replace("\n", " ")
        txt = articles['summary'].replace("\n", " ")
        if len(articles['author']) == 1:
            a = articles['author']["name"]
        else:
            a = articles['author'][0]['name']

        if a not in id2aut:
            id2aut[a] = Author(a)

        id2doc[f"arx{key}"] = Document(title, id2aut[a], articles['published'], articles['id'], txt)
        key+=1



def main():
    reddit()
    arxiv()
    print(id2doc)
    # create dataframe
    # df = pd.DataFrame(corpus)
    # print(df)



if __name__ == '__main__':
    main()
