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
from flaskexample.load_examples import load_example

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

@app.route('/examples')
def poeml_examples():
    return render_template("examples.html")

@app.route('/ex1')
def poeml_ex1():
    df_image,df_poems = load_example('ex1')
    return render_template("ex1.html",df_image=df_image,df_poems=df_poems)

@app.route('/ex2')
def poeml_ex2():
    df_image,df_poems = load_example('ex2')
    return render_template("ex2.html",df_image=df_image,df_poems=df_poems)

@app.route('/algorithms')
def poeml_algorithms():
    return render_template("algorithms.html")

@app.route('/about')
def poeml_about():
    return render_template("about.html")

@app.route('/contact')
def poeml_contact():
    return render_template("contact.html")
