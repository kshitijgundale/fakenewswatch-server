from tweet_scraper.scraper import TweetsScraper
from googlesearch import search
from newspaper import Article, ArticleException
from app.fakenews.propagation_graph import PropagationGraph
from app.fakenews.utils import get_account_age, followers_count, following_count, is_user_verified,\
    num_hashtags, num_user_mentions, node_type, get_vader_score, time_diff_with_source
from torch_geometric.loader import DataLoader
from app.fakenews.gnn import GNN
import torch
import uuid

model_data = torch.load("./app/fakenews/politifact_model.pt")
model_params = model_data['model_params']
state_dict = model_data['state_dict']

model = GNN(**model_params)
model.load_state_dict(state_dict)

def link_preview(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            'id': uuid.uuid4(),
            'url': url,
            'title': article.title,
            'image': article.top_image,
            'text': str(article.text[:100]) + "..."
        }
    except ArticleException:
        return None
        
def recommend_news(statements, urls):
    news = []
    for s in statements:
        news_cnt = 0
        for i in search(s, tld="co.in", num=10, stop=None, pause=2):
            if news_cnt == 5:
                break
            news.append(link_preview(i))
            news_cnt += 1

    for u in urls:
        try:
            news_cnt = 0
            article = Article(u)
            article.download()
            article.parse()
            if article.title:
                for i in search(article.title, tld="co.in", num=10, stop=None, pause=2):
                    if news_cnt == 5:
                        break
                    news.append(link_preview(i))
                    news_cnt += 1
        except ArticleException:
            continue

    return {"news": news}

def get_twitter_data(
    statements, urls, feedback,
    num_news, num_tweets,
    inlcude_retweets, include_quotes, include_replies
):
    if feedback:
        urls += feedback
        get_related_news_urls_from_statement = False
        get_related_news_urls_from_urls = False
    else:
        get_related_news_urls_from_statement = True
        get_related_news_urls_from_urls = True
    
    scraper = TweetsScraper(
        statements=statements,
        urls=urls,
        since="2019-01-01",
        until="2022-01-01",
        url_search=True,
        keywords_from_statement=True,
        get_related_news_urls_from_statement=get_related_news_urls_from_statement,
        get_related_news_urls_from_urls=get_related_news_urls_from_urls,
        num_news=num_news,
        max_tweets=num_tweets,
        fetch_retweets=inlcude_retweets,
        fetch_quotes=include_quotes,
        fetch_replies=include_replies,
        APP_KEY="ASjMzMTXZIG8oqevrZeTU7G05",
        APP_SECRET = "EaVgcUQvNPCzFbZtNDhnvC4aIIY0yrVrFReHXVyCead9kyvDer"
    )

    data = scraper.get_twitter_data()

    if not data['tweets']:
        return []

    graph = PropagationGraph(
        tweets = data['tweets'],
        retweets = data['retweets'],
        quotes = None,
        replies= None,
        features=[
            "account_age", "is_verified", "follower_count", "following_count",
            "num_hastags", "num_user_mentions", "node_type", "source_time_diff", "vader_score"
        ], 
        feature_extractors=[
            get_account_age, followers_count, following_count, is_user_verified,
            num_hashtags, num_user_mentions, node_type, get_vader_score, time_diff_with_source
        ], 
        deadline = None,
        news_utc = None,
        news_id = "test"
    )

    pyg_graph = graph.get_pyg_data_object()
    nodes, edges = graph.get_visjs_data()

    pred = DataLoader([pyg_graph])
    result = None
    with torch.no_grad():
        for i in pred:
            result = torch.softmax(model(i.x, i.edge_index, i.batch), dim=1)[0][1].item() * 100

    return {
        "nodes": nodes,
        "edges": edges,
        "result": result,
        "metadata": graph.metadata
    }