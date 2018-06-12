#!/usr/bin/env python

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd

username = 'kehrl'
dbname_craigslist = 'craigslist_db'
dbname_ebay = 'ebay_db'

engine_craigslist = create_engine('postgres://%s@localhost/%s'%(username,dbname_craigslist))
engine_ebay = create_engine('postgres://%s@localhost/%s'%(username,dbname_ebay))
print(engine_craigslist.url)
print(engine_ebay.url)

if not(database_exists(engine_craigslist.url)):
    create_database(engine_craigslist.url)
if not(database_exists(engine_ebay.url)):
    create_database(engine_ebay.url)

craigslist_postings = pd.read_csv('data/seattle_craigslist_postings.csv', index_col = 0, sep = ',')
ebay_postings = pd.read_csv('data/ebay_postings.csv', index_col = 0, sep = ',')

craigslist_postings.to_sql('craigslist_postings', engine_craigslist, if_exists='replace')
ebay_postings.to_sql('ebay_postings', engine_ebay, if_exists='replace')