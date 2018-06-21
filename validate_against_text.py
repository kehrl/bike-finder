#!/usr/bin/env python 

from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import numpy as np
import keras
from flasklib import bikelib
import os

user = 'kehrl'                   
host = 'localhost'

# Databases
dbname = 'bike_db'
engine = create_engine('postgres://%s@localhost/%s'%(user,dbname))

# Connect to SQL databases and pull data
con = psycopg2.connect(database = dbname, user = user)
labeled_df = pd.read_sql("SELECT * FROM training_postings WHERE website='craigslist'", con=con)
con.close()

# Bike labels
bike_labels = list(np.load("weights/train_labels.npy")[()])

# Get only labeled craigslist postings in test set
ind = []
for i in range(0,len(labeled_df)):
    if not(os.path.isfile('data/test/'+\
            labeled_df.biketype.iloc[i]+'/'+\
            os.path.split(labeled_df.imagefile.iloc[i])[-1])):
        ind.append(i)
labeled_df.drop(index=ind, inplace=True)

def get_text_labels(labeled_df):

    labeled_df['textlabel'] = ''
    for biketype in labeled_df.biketype.unique():
        labeled_df.loc[labeled_df.title.str.contains(biketype), 'textlabel'] = biketype  

    return labeled_df
  
def get_picture_labels(labeled_df, top_model_path='model.best.hdf5'):

    # Load top model
    model = keras.models.load_model(top_model_path)    

    # Get bottleneck features
    image_files = np.array(labeled_df['imagefile'])
    bottleneck_features = bikelib.get_bottleneck_features(image_files)

    # Get predictions
    preds = model.predict(bottleneck_features)
    labeled_df['imagelabel'] = [bike_labels[i] for i in np.argmax(preds, axis = -1)]

    return labeled_df

def get_categorical_accuracy(true_labels, predicted_labels, labels):
    
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    for label in labels:
        if label != '':
            tp += len(np.where([(true_labels == label) & (predicted_labels == label)])[1])
            fp += len(np.where([(true_labels != label) & (predicted_labels == label)])[1])    
            tn += len(np.where([(true_labels != label) & (predicted_labels != label)])[1])    
            fn += len(np.where([(true_labels == label) & (predicted_labels != label)])[1])    
    
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    precision = tp / (tp + fp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    return accuracy, precision, tpr, tnr


# Get labels based on text  
labeled_df = get_text_labels(labeled_df)

# Get labels based on images
labeled_df = get_picture_labels(labeled_df)

# Get stats for the two approaches to categorizing bikes
labels = ['road', 'mountain', 'kids', 'bmx', 'cruiser', 'tandem']
acc_text, prec_text, tpr_text, tnr_text = get_categorical_accuracy(labeled_df['biketype'], labeled_df['textlabel'], labels)

acc_image, prec_image, tpr_image, tnr_image = get_categorical_accuracy(labeled_df['biketype'], labeled_df['imagelabel'], labels) 