from flask import request
from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import random
import spacy
import numpy as np
from flaskexample.a_Model import ModelIt


@app.route('/')

@app.route('/index')
def index():
    return render_template("input.html")

@app.route('/input')
def poeml_input():
    return render_template("input.html")

@app.route('/output')
def poeml_output():
  url = request.args.get('key_words')
  best_matches = ModelIt(url)
  return render_template("output.html", best_matches=best_matches)
