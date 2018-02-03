from flask import Flask
app=Flask(__name__)
app._static_folder = "/home/ubuntu/app_demo/flaskexample/static"
from flaskexample import views
