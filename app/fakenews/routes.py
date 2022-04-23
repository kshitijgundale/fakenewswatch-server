from app.fakenews import bp
from flask import request, jsonify
from app.fakenews import service
from flask_cors import cross_origin

@bp.route('/prediction', methods=['POST'])
def get_predictions():
    request_data = request.get_json()
    statements = request_data['statements']
    urls = request_data['urls']
    feedback = request_data['feedback']

    num_tweets = int(request_data['num_tweets'])
    num_news = int(request_data['num_news'])
    include_retweets = request_data['include_retweets']
    include_quotes = request_data['include_quotes']
    include_replies = request_data['include_replies']

    return jsonify(service.get_twitter_data(
        statements, urls, feedback,
        num_news, num_tweets,
        include_retweets, include_quotes, include_replies
    ))

@bp.route('/recommend', methods=['POST'])
@cross_origin()
def get_news_recommendation():
    request_data = request.get_json()
    statements = request_data['statements']
    urls = request_data['urls']

    return jsonify(service.recommend_news(statements, urls))




