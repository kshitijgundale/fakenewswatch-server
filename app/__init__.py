from flask import Flask
from app.fakenews import bp as fakenews_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.debug = True

app.register_blueprint(fakenews_bp, url_prefix='/fakenews')
