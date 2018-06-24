#!/usr/bin/env python

'''
Save current Craigslist postings to SQL database bike_db.

Laura Kehrl
'''

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd

username = open('postgres_user','r').readlines()[0].split()[0]
dbname = 'bike_db'

engine = create_engine('postgres://%s@localhost/%s'%(username,  dbname))

if not(database_exists(engine.url)):
    create_database(engine.url)

craigslist_postings_current = pd.read_csv('data/seattle_craigslist_postings_current.csv', \
          index_col = 0, sep = ',')

# Do some data cleanup
craigslist_postings_current['imagefile'] = craigslist_postings_current['imagefile'].str.strip()

# Remove duplicates
craigslist_postings_current = craigslist_postings_current.drop_duplicates(subset = 'URL')

craigslist_postings_current.to_sql('current_craigslist_postings', engine, if_exists='replace')