#!/usr/bin/env python

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd

username = 'kehrl'
dbname = 'bike_db'

engine = create_engine('postgres://%s@localhost/%s'%(username,  dbname))

if not(database_exists(engine.url)):
    create_database(engine.url)

craigslist_postings = pd.read_csv('data/seattle_craigslist_postings.csv', index_col = 0, sep = ',')
craigslist_postings_current = pd.read_csv('data/seattle_craigslist_postings_current.csv', \
          index_col = 0, sep = ',')
ebay_postings = pd.read_csv('data/ebay_postings.csv', index_col = 0, sep = ',')

# Do some data cleanup
for postings in [craigslist_postings, ebay_postings]:
    postings['imagefile'] = postings['imagefile'].str.strip()

# Remove duplicates
craigslist_postings_current = craigslist_postings_current.drop_duplicates(subset = 'URL')

craigslist_postings.to_sql('all_craigslist_postings', engine, if_exists='replace')
craigslist_postings_current.to_sql('current_craigslist_postings', engine, if_exists='replace')
ebay_postings.to_sql('all_ebay_postings', engine, if_exists='replace')