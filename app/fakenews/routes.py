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

    return jsonify(service.get_twitter_data(urls, statements, feedback))

@bp.route('/recommend', methods=['POST'])
@cross_origin()
def get_news_recommendation():
    request_data = request.get_json()
    statements = request_data['statements']
    urls = request_data['urls']

    return jsonify(service.recommend_news(statements, urls))




