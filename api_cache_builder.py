import pickle

import praw
import urllib3
import xmltodict

def generate():
    reddit_client = praw.Reddit(client_id='T9qPLZ2H3esl_ukXzn0UVA', client_secret='zyGz9sEs67D0LB3Ihu3kia_3oq0KlQ',
                                user_agent='search_engine_td')
    # todo sbr title should be a param
    reddit_dict = reddit_client.subreddit("france").hot(limit=10)

    resp = urllib3.request('GET', 'http://export.arxiv.org/api/query?search_query=all:electron&max_results=10')
    arxiv_dict = xmltodict.parse(resp.data)

    dd = reddit_dict | arxiv_dict

    print(dd)
    with open("out.pkl", "wb") as f:
        pickle.dump(dd, f)

    with open("out.pkl", "rb") as f:
        dd = pickle.load(f)

    print(dd)

if __name__ == '__main__':
    generate()
