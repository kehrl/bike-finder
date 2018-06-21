#!/usr/bin/env python 

import pandas as pd
import numpy as np
import psycopg2
import os
import shutil
# import imagelib

# Inputs
n_train = 2000 # per bike category
n_test = 75

# Output dirs
main_dir = '/Users/kehrl/Code/bike-finder/data/'
train_data_dir = main_dir+'train/'
test_data_dir  = main_dir+'test/'

def save_images_by_category(category, labeled_data, data_dir, train_dir, test_dir, \
          n_train = 200, n_test = 20):
    
    # Create directories
    os.makedirs(train_data_dir+category)
    os.makedirs(test_data_dir+category)
    
    # Subset data based on category    
    category_data = labeled_data[labeled_data.biketype == category]
    
    # Number of elements in category
    n_category = len(category_data)
    
    # Generate training and testing data
    # If more data is available than needed, just pull random data.
    if n_category > n_train + n_test:
        # Prioritize craigslist
        ind_craigslist = np.where(category_data['website'] == 'craigslist')[0]
        ind_ebay = np.where(category_data['website'] == 'ebay')[0]
        
        # Get indices from craigslist
        if len(ind_craigslist) < n_train + n_test:
            ind = ind_craigslist.copy()
            ind = np.r_[ind, np.random.choice(ind_ebay, n_train + n_test - len(ind), \
                    replace = False)]
        else:
            ind = np.random.choice(ind_craigslist, n_train + n_test, replace = False)
        
        # Check to make sure there are no repeating indices
        if not(len(np.unique(ind)) == len(ind)):
            print("Repeating indices, check code")
        
        ind = np.random.permutation(ind)
        ind_train = ind[0:n_train]
        ind_test = ind[n_train:]
        
        # Get images
        images_train = category_data['imagefile'].iloc[ind_train]
        images_test = category_data['imagefile'].iloc[ind_test]
       
        
    # If less data is available than needed, pull all data and generate additional images.    
    else:
        # Get available data
        images = category_data['imagefile']
       
        # Get craigslist postings for testing
        ind_craigslist = np.where(category_data['website'] == 'craigslist')[0]
        ind_ebay = np.where(category_data['website'] == 'ebay')[0]
       
        # Generate additional data
        # Calculate number of additional images needed
        n_sample_train = int(n_train/(n_train + n_test) * n_category)
       
        # Get images that will be used for generating additional data
        if len(ind_craigslist) < n_test:
            ind_test = np.r_[ind_craigslist, np.random.choice(ind_ebay, \
                n_test - len(ind_craigslist), replace = False)]
        else:
            ind_test = np.random.choice(ind_craigslist, n_test, replace = False)
        ind_sample_train = np.setdiff1d(np.arange(n_category), ind_test)
       
        # Generate duplicate images, which will be modified later in train_model.py by 
        # ImageDataGenerator
        ind_train = np.random.choice(ind_sample_train, n_train)
       
        # Get images (including repeating ones)
        images_train = category_data['imagefile'].iloc[ind_train]
        images_test = category_data['imagefile'].iloc[ind_test]
       
    # Save images
    image_file_previous = []
    
    n = 0
    for image_file in np.sort(images_train):
        image_name = os.path.split(image_file)
        # If image is already present in directory, we need to save it with a new name.
        if image_file == image_file_previous:
            shutil.copy(data_dir+image_file, train_dir+category+'/'+image_name[-1][0:-4]+\
                    '_'+str(n)+'.jpg')
            n += 1
        # Otherwise, just save image.
        else:
            shutil.copy(data_dir+image_file, train_dir+category+'/'+image_name[-1])
            n = 0
            image_file_previous = str(image_file)
    
    n = 0
    for image_file in np.sort(images_test):
        # If image is already present in directory, we need to save it with a new name.
        image_name = os.path.split(image_file)
        if image_file == image_file_previous:
            shutil.copy(data_dir+image_file, test_dir+category+'/'+image_name[-1][0:-4]+\
                    '_'+str(n)+'.jpg')
            n += 1
        else:
            # Otherwise, just save image.
            shutil.copy(data_dir+image_file, test_dir+category+'/'+image_name[-1])
            n = 0
            image_file_previous = str(image_file)
        
    return images_train, images_test


###########################################################################

if __name__ == "__main__":

    # Get data from SQL database
    con = psycopg2.connect(database = 'bike_db', user = 'kehrl')
    labeled_data = pd.read_sql('SELECT * FROM training_postings', con = con)

    # Get bike categories
    categories = labeled_data.biketype.unique()

    # Check if train and test dirs exist
    # if not, create the dirs; if so, delete contents for re-generation
    if os.path.exists(train_data_dir):
        shutil.rmtree(train_data_dir)
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
    os.makedirs(train_data_dir)
    os.makedirs(test_data_dir)
    
    # Copy images to train and test directories
    images_train = {}
    images_test = {}
    for category in categories:
        images_train[category], images_test[category] = save_images_by_category(category, \
                labeled_data, main_dir, train_data_dir, test_data_dir, n_train, n_test)
    