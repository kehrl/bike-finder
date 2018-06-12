#!/usr/bin/env python 

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib

user = 'kehrl'                   
host = 'localhost'

# Categories
bike_labels = ['not bike','road bike', 'mountain bike', 'tandem', 'bmx', 'cruiser', \
               'kids', 'hybrid','triathlon','other']

# Databases
dbname_ebay = 'ebay_db'                         # Pulling data
dbname_craigslist = 'craigslist_db'   # Pulling data
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
n_craigslist = {} # Save number of bikes labeled based on different categories
n_ebay = {}
for label in bike_labels:

    # Label ebay bikes by provided biketype
    if not(label in ['not bike', 'other']):
        ebay_labeled.loc[ebay_unlabeled.biketype.str.contains(label, na=False),\
                'biketype'] = label
    
    sum_ebay = ebay_unlabeled.biketype.str.contains(label, na=False).sum()
    sum_craigslist = 0
    
    for labeled, n_count, sum_label in zip([ebay_labeled, craigslist_labeled], \
            [n_ebay, n_craigslist], [sum_ebay, sum_craigslist]): 
        
        if label == 'not bike':
            # Search for key terms that describe "not bike" in item description and title
            notbike_labels = ['exercise bike', 'exercise bicycle', 'shoes'] #scooter
            notbike_labels = ['shoes']    
            
            for notbike_label in notbike_labels:
                labeled.loc[labeled.description.str.contains(notbike_label, na=False),'biketype'] = label  
                labeled.loc[labeled.title.str.contains(notbike_label, na=False),'biketype'] = label    

        elif not(label == 'other'):
            # Search for label in item description
            labeled.loc[labeled.description.str.contains(label, na=False),'biketype'] = label
    
            # Search for label in titles
            labeled.loc[labeled.title.str.contains(label, na=False),'biketype'] = label
                    
        # Sum number of labeled postings for pie charts
        n_count[label] = sum_label + labeled.title.str.contains(label, na=False).sum() + \
            labeled.title.str.contains(label, na=False).sum()

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
for i in range(len(bike_labels)-2):
    values_craigslist[i] = n_craigslist[bike_labels[i]]
    values_ebay[i] = n_ebay[bike_labels[i]]
values_craigslist[-1] = len(craigslist_labeled) - values_craigslist.sum()
values_ebay[-1] = len(ebay_labeled) - values_ebay.sum()

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
    

        

    

