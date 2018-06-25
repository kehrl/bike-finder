#!/usr/bin/env python 

'''
Predict bike types for Craigslist postings and update 
database.

Laura Kehrl
'''

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import requests

import keras
from keras.applications import InceptionV3
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

# Cutoff for identifying bikes for softmax classifier
cutoff = 0.5 

user = open('postgres_user','r').readlines()[0].split()[0]                   
passwd = open('postgres_passwd','r').readlines()[0].split()[0]
host = 'localhost'

# Data directory
main_data_dir = 'data/'

# Model inputs
model_name = 'inception'

if model_name == 'vgg16':
    image_shape = (224, 224)
elif model_name == 'inception':
    image_shape = (299, 299)
top_model_path = 'model.best.hdf5'
bike_labels = np.sort(list(np.load("weights/train_labels.npy")[()]))

# Get current postings
dbname = 'bike_db'
engine = create_engine('postgres://%s:%s@localhost/%s'%(user, passwd, dbname))
con = psycopg2.connect(database=dbname, user=user, password=passwd)
current_postings = pd.read_sql('SELECT * FROM current_craigslist_postings', con=con)
old_predicted_postings = pd.read_sql('SELECT * FROM predicted_postings', con=con)
con.close()

# Get files for prediction
current_postings = current_postings.drop(columns=['biketype'])
current_predicted_postings = current_postings.copy()
current_predicted_postings = pd.merge(current_postings[['title','price', 'URL', 'imageURL', 'imagefile', 'description', 'timeposted', 'latitude', 'longitude']],
        old_predicted_postings[['URL','biketype', 'match']], how='left', on=['URL'])
current_predicted_postings.drop_duplicates(subset='URL', inplace=True)
ind_unlabeled = np.where(current_predicted_postings.biketype.isnull())[0]
print(len(ind_unlabeled))
if len(ind_unlabeled) > 200:
    ind_unlabeled = ind_unlabeled[0:200]
image_files = np.array(current_predicted_postings.loc[ind_unlabeled,'imagefile'])

# Dictionary of models
model_options = {
	"vgg16": VGG16,
	"inception": InceptionV3,
}

# SQL database for saving predicted values
def get_bottleneck_features(network, image_files, image_shape):
    '''
    bottleneck_features = get_bottleneck_features(network, image_files, 
          image_shape)
    
    Get bottleneck features for the last max pool layer of trained CNN.
    '''
    
    print('Getting bottleneck features...')

    model = network(weights="imagenet", include_top=False)

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        fill_mode='nearest'
        )

    images = np.zeros([len(image_files), image_shape[0], image_shape[0], 3]) 
    for i in range(len(image_files)):
        image = load_img('data/'+image_files[i].strip(), target_size=image_shape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        images[i] = image
    
    bottleneck_features = model.predict_generator(datagen.flow(images, shuffle=False, batch_size=10), verbose=1)
        
    return bottleneck_features

###############################################################################

if __name__ == "__main__":

    # Load top model
    model = keras.models.load_model(top_model_path)

    # Get bottleneck features
    bottleneck_features = get_bottleneck_features(model_options[model_name], image_files, \
        image_shape)

    preds = model.predict(bottleneck_features)

    current_predicted_postings.loc[ind_unlabeled, 'biketype'] = [bike_labels[i] for i in np.argmax(preds, axis = -1)]
    current_predicted_postings.loc[ind_unlabeled, 'match'] = np.round(np.max(preds, axis = -1)*100)
    
    current_predicted_postings.loc[current_predicted_postings.match < cutoff*100, 'biketype'] = ''

    # Save as new table
    current_predicted_postings.to_sql('predicted_postings', engine, if_exists='replace')
