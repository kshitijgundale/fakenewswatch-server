from flask import Blueprint

bp = Blueprint('fakenews', __name__)

from app.fakenews import routes