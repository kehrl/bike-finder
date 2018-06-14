#!/usr/bin/env python 

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib
import os

user = 'kehrl'                   
host = 'localhost'

# Categories
bike_labels = ['road bike', 'mountain bike', 'not bike', 'kids'] # not bike, tandem, triathlon, hybrid, bmx, kids bike, cruiser bike

# Labeled data directory
data_labeled_dir = '/Users/kehrl/Code/bike-finder/data/labeled/'

# Databases
dbname_ebay = 'ebay_db'                         # Pulling data
dbname_craigslist = 'craigslist_db'             # Pulling data
dbname_labeled = 'labeled_db'                   # Pushining data

engine_labeled = create_engine('postgres://%s@localhost/%s'%(user,dbname_labeled))
if not(database_exists(engine_labeled.url)):
    create_database(engine_labeled.url)

# Connect to SQL databases and pull data
con_ebay = psycopg2.connect(database = dbname_ebay, user = user)
con_craigslist = psycopg2.connect(database = dbname_craigslist, user = user)

craigslist_unlabeled = pd.read_sql('SELECT * FROM craigslist_postings', con = con_craigslist)
ebay_unlabeled = pd.read_sql('SELECT * FROM ebay_postings', con = con_ebay)
   
# Use biketypes, titles, and posting descriptions to label bikes
for unlabeled in [ebay_unlabeled, craigslist_unlabeled]:
    for column in ['biketype', 'description', 'title']:
        unlabeled[column] = unlabeled[column].str.lower()
ebay_labeled = ebay_unlabeled.copy(deep = True)
ebay_labeled.biketype = ''

craigslist_labeled = craigslist_unlabeled.copy(deep = True)
craigslist_labeled.biketype = ''

# Label bikes
for label in bike_labels:
    
    files = os.listdir(data_labeled_dir+label)
    for file in files:
        if 'craigslist' in file:
            ind = np.where(craigslist_labeled.imagefile == file)
            craigslist_labeled.biketype.iloc[ind] = label          
        else:
            ind = np.where(ebay_labeled.imagefile == file)
            ebay_labeled.biketype.iloc[ind] = label    
        
# Concat labeled craigslist and ebay bikes into one pandas data frame
craigslist_labeled['website'] = 'craigslist'
ebay_labeled['website'] = 'ebay' 
labeled_data = pd.concat([craigslist_labeled, ebay_labeled])

# Remove all rows with unlabeled bikes
labeled_data = labeled_data[labeled_data.biketype.map(len) > 0]

# Load all labeled bikes into SQL database for model training
labeled_data.to_sql('labeled_data', engine_labeled, if_exists='replace')

# Make pie charts showing data distribution
matplotlib.rc('font',family='sans-serif',size=10)

values_craigslist = np.zeros([len(bike_labels),])
values_ebay = np.zeros([len(bike_labels),])
for i in range(len(bike_labels)):
    if not(bike_labels[i] == 'other'):
        values_craigslist[i] = craigslist_labeled.groupby('biketype').size()[bike_labels[i]]
        values_ebay[i] = ebay_labeled.groupby('biketype').size()[bike_labels[i]]
    else:
        values_craigslist[i] = craigslist_labeled.groupby('biketype').size()['']
        values_ebay[i] = ebay_labeled.groupby('biketype').size()['']    

# Ebay plot
plt.figure()
total = sum(values_ebay)    
plt.pie(values_ebay, labels = bike_labels, \
      autopct = lambda p: '{:.0f}'.format(p * total / 100))

plt.axis('equal')
plt.savefig('figures/ebay_training_data.pdf')
plt.close()

# Craigslist plot
plt.figure()
total = sum(values_craigslist)    
plt.pie(values_craigslist, labels = bike_labels, \
      autopct = lambda p: '{:.0f}'.format(p * total / 100))
plt.axis('equal')
plt.savefig('figures/craigslist_training_data.pdf')
plt.close()
    

        

    

