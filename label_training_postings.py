#!/usr/bin/env python 

'''
Create SQL table of labeled postings for trainings. The label is taken from
the folder where the image is located.

Laura Kehrl
'''

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import matplotlib
import os

user = open('postgres_user','r').readlines()[0].split()[0]                  
host = 'localhost'

# Categories
bike_labels = ['bmx', 'road', 'mountain', 'cruiser', 'kids', 'other', 'tandem'] # not bike, tandem, triathlon, hybrid, bmx, kids bike, cruiser bike

# Labeled data directory
data_labeled_dir = '/Users/kehrl/Code/bike-finder/data/labeled/'

# Databases
dbname = 'bike_db'
engine = create_engine('postgres://%s@localhost/%s'%(user,dbname))

# Connect to SQL databases and pull data
con = psycopg2.connect(database = dbname, user = user)
craigslist_unlabeled = pd.read_sql('SELECT * FROM all_craigslist_postings', con = con)
ebay_unlabeled = pd.read_sql('SELECT * FROM all_ebay_postings', con = con)
con.close()
   
# Use biketypes, titles, and posting descriptions to label bikes
for unlabeled in [ebay_unlabeled, craigslist_unlabeled]:
    for column in ['biketype', 'description']:
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
            craigslist_labeled.imagefile.iloc[ind] = 'labeled/'+label+'/'+file          
        elif 'ebay' in file:
            ind = np.where(ebay_labeled.imagefile == file)
            ebay_labeled.biketype.iloc[ind] = label
            ebay_labeled.imagefile.iloc[ind] = 'labeled/'+label+'/'+file
        elif file.endswith('.jpg'):
            ebay_labeled = ebay_labeled.append({'imagefile': 'labeled/'+label+'/'+file, 'biketype': label}, ignore_index = True)
          
# Concat labeled craigslist and ebay bikes into one pandas data frame
craigslist_labeled['website'] = 'craigslist'
ebay_labeled['website'] = 'ebay' 
labeled_data = pd.concat([craigslist_labeled, ebay_labeled])

# Remove all rows with unlabeled bikes
labeled_data = labeled_data[labeled_data.biketype.map(len) > 0]

# Load all labeled bikes into SQL database for model training
labeled_data.to_sql('training_postings', engine, if_exists='replace')

# Make pie charts showing data distribution
matplotlib.rc('font',family='sans-serif',size=20)
matplotlib.rcParams['text.color'] = 'w'

values = np.zeros([len(bike_labels),])
for i in range(len(bike_labels)):
    values[i] = labeled_data.groupby('biketype').size()[bike_labels[i]]


plt.figure()
total = sum(values)    
plt.pie(values, labels = ['bmx', 'road', 'mountain', 'cruiser', 'kids', 'other', 'tandem'], \
      autopct = lambda p: '{:.0f}'.format(p * total / 100))

plt.axis('equal')
plt.savefig('figures/labeled_data.png', format='PNG', transparent = True)
plt.close()
        

    

