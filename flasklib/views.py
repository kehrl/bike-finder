from werkzeug.utils import secure_filename
from keras.applications import InceptionV3
from keras.preprocessing.image import img_to_array, load_img
import keras
import numpy as np
from flask import render_template
from flask import request
from flasklib import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import tensorflow as tf
import datetime
import psycopg2
import os

user = open('postgres_user','r').readlines()[0].split()[0]          
passwd = passwd = open('postgres_passwd','r').readlines()[0].split()[0]
host = 'localhost'
dbname = 'bike_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database=dbname, user=user, password=passwd)

graph = tf.get_default_graph()
model = InceptionV3(weights='imagenet', include_top=False)
top_model = keras.models.load_model('model.best.hdf5')
bike_labels=np.sort(list(np.load("weights/train_labels.npy")[()]))

@app.route('/')
@app.route('/index')
def index():
        
    return render_template("index.html",
            title = 'Home', user = { 'nickname': 'kehrl' })

@app.route('/db')
def bike_page():
    sql_query = """                                                             
                SELECT * FROM predicted_postings \
;                                                                               
                """
    global dbname, user, passwd
    con = psycopg2.connect(database=dbname, user=user, password=passwd)
    query_results = pd.read_sql_query(sql_query,con)
    con.close

    bikes = ""
    print(query_results[:10])
    for i in range(0,10):
        bikes += query_results.iloc[i]['biketype']
        bikes += "<br>"
    return bikes

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = 'SELECT title, biketype, price, match, timeposted, "imageURL", "URL" FROM predicted_postings ORDER BY match DESC, timeposted DESC;'
    
    global dbname, user, passwd
    con = psycopg2.connect(database=dbname, user=user, password=passwd)
    query_results = pd.read_sql_query(sql_query,con)
    con.close()

    bikes = []
    for i in range(0,query_results.shape[0]):
        bikes.append(dict(title=query_results.iloc[i]['title'], \
        imageURL=query_results.iloc[i]['imageURL'], URL=query_results.iloc[i]['URL'], \
        price=query_results.iloc[i]['price']))
    return render_template('bikes.html',bikes=bikes)

@app.route('/input')
def cesareans_input():
    return render_template("input.html")
    
@app.route('/about')
def about():
    return render_template("bikes.html")

@app.route('/output', methods=['POST','GET'])
def cesareans_output():
    # Pull 'biketype' and 'searchterm' from input field and store it
    
    if request.method == 'POST':
        biketype = request.form.get('biketype')
        searchterm = request.form.get('searchterm')
           
        # Get image
        image_file = request.files.get('file')
        if image_file:
            image_name = secure_filename(image_file.filename)
        
            image_path = 'uploads/'+image_name
            image_file.save(image_path)        
            
            image = load_img(image_path, target_size=(299, 299))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)*(1./255)
            
            global graph, bike_labels
            with graph.as_default():
                pred = model.predict(image)
                pred = top_model.predict(pred)
            biketype = bike_labels[np.argmax(pred)]
    
    if biketype == 'None selected':
        error_message = "A bike type or image file must be selected."
    elif biketype == 'other':
        error_message = "I can't find any matches. Please try another bike image."
        biketype = 'None selected'
    else:
        error_message = ''
    
    #Select the desired bike type from the bike database
    query = 'SELECT title, biketype, price, match, description, timeposted, imagefile, "imageURL", "URL" FROM predicted_postings WHERE biketype=%s ORDER BY match DESC, timeposted DESC;' % (str("'"+biketype+"'"))

    global dbname, user, passwd
    con = psycopg2.connect(database=dbname, user=user, password=passwd)
    query_results = pd.read_sql_query(query,con)
    con.close()    

    # Set up search term criterion 
    if searchterm.startswith('"') and searchterm.endswith('"'):
        terms = [searchterm[1:-1].lower()]
    else:
        terms = searchterm.lower().split()

    bikes = []
    for i in range(0,query_results.shape[0]):
        match_search_term = True
        for term in terms:
            if (term in query_results.iloc[i]['title'] or term in query_results.iloc[i]['description']) and (match_search_term == True):
                match_search_term = True
            else:
                match_search_term = False
        if (searchterm == '') or match_search_term == True:
            timeposted = datetime.datetime.strptime(query_results.iloc[i]['timeposted'], ' %Y-%m-%d %I:%M%p')
            bikes.append(dict(title=query_results.iloc[i]['title'], \
                imageURL=query_results.iloc[i]['imageURL'], \
                price=query_results.iloc[i]['price'], \
                URL=query_results.iloc[i]['URL'], \
                imagefile=query_results.iloc[i]['imagefile'], \
                timeposted=timeposted.strftime('%Y-%m-%d %H:%M'), \
                match=int(np.round(query_results.iloc[i]['match']))))
    n_bikes = len(bikes)
    return render_template("output.html", bikes = bikes, n_bikes = n_bikes, biketype_final = biketype, error_message = error_message)

