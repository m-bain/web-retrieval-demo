from flask import Flask
from demo.config import Config

app = Flask(__name__)
app.config.from_object(Config)

from demo.app import routes
