from flask import render_template
from flask import request
from flasklib import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flasklib.a_Model import ModelIt

user = 'kehrl' #add your username here (same as previous postgreSQL)            
host = 'localhost'
dbname = 'bike_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'kehrl' },
       )

@app.route('/db')
def bike_page():
    sql_query = """                                                             
                SELECT * FROM predicted_postings WHERE website='craigslist'\
;                                                                               
                """
    query_results = pd.read_sql_query(sql_query,con)
    bikes = ""
    print(query_results[:10])
    for i in range(0,10):
        bikes += query_results.iloc[i]['biketype']
        bikes += "<br>"
    return bikes

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = 'SELECT title, biketype, price, match, "imageURL", "URL" FROM predicted_postings WHERE website=%s ORDER BY match DESC;' % str('''craigslist''')
    query_results=pd.read_sql_query(sql_query,con)
    bikes = []
    for i in range(0,query_results.shape[0]):
        bikes.append(dict(title=query_results.iloc[i]['title'], \
        imageURL=query_results.iloc[i]['imageURL'], URL=query_results.iloc[i]['URL'], \
        biketype=query_results.iloc[i]['price']))
    return render_template('bikes.html',bikes=bikes)

@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/output')
def cesareans_output():
    #pull 'biketype' from input field and store it
    biketype = request.args.get('biketype')
    #just select the desired bike type from the bike database
    query = 'SELECT title, biketype, price, match, "imageURL", "URL" FROM predicted_postings WHERE website=%s AND biketype=%s ORDER BY match DESC;' % (str("'craigslist'"), str("'"+biketype+"'"))
    print(query)
    query_results=pd.read_sql_query(query,con)
    print(query_results)
    bikes = []
    for i in range(0,query_results.shape[0]):
        bikes.append(dict(title=query_results.iloc[i]['title'], \
              imageURL=query_results.iloc[i]['imageURL'], \
              price=query_results.iloc[i]['price'], \
              URL=query_results.iloc[i]['URL']))
    the_result = 0 #ModelIt(patient,bikes)
    return render_template("output.html", bikes = bikes, the_result = the_result)

