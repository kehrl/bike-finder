#!/usr/bin/env python

import ebaysdk
from ebaysdk.finding import Connection as Finding
from ebaysdk.exception import ConnectionError
from ebaysdk.shopping import Connection as Shopping
import numpy as np
import urllib
import os
from collections import defaultdict

def get_ebay_postings(city):

    api = Finding(config_file='ebay.yaml')

    n_pages = 5
    posting_ids = []
    for n_page in range(n_pages):
        api_request = {
            'itemFilter': [
            {'name': 'HideDuplicateItems', 'value': True}],
            'paginationInput': {'entriesPer': 100, 'pageNumber': n_page + 1},
            'sortOrder': 'StartTimeNewest',
            'categoryId': '177831',
            'Country': 'US', 'ListingStatus': 'Active'}
        response = api.execute('findItemsAdvanced', api_request)
    
        items = response.reply.searchResult.item
        for item in items:
            posting_ids.append(item.itemId)
          
    return posting_ids

def check_against_saved_postings(posting_ids):

    # Open csv file 
    if os.path.exists('data/'+'ebay_postings.csv'):
        fid = open('data/'+'ebay_postings.csv','r')
        lines = fid.readlines()
    
        # Get saved itemIDs
        saved_posting_ids = []
        for line in lines[1:]:
            saved_posting_ids.append(line.split(',')[0].replace(' ',''))
        fid.close()
 
        # Find postings that haven't been saved in the directory already
        new_posting_ids = []   
        for posting_id in posting_ids:
            if not(posting_id in saved_posting_ids):
                new_posting_ids.append(posting_id)  
            
        # Find postings that have been deleted from Craigslist      
        deleted_posting_ids = []
        for posting_id in saved_posting_ids:
            if not(posting_id in posting_ids):
                deleted_posting_ids.append(posting_id)
    else:
      new_posting_ids = posting_ids
      deleted_posting_ids = []

    return new_posting_ids, deleted_posting_ids
    
    
def get_new_posting_attrs(posting_ids):

    n_items = len(posting_ids)

    bike_attrs = defaultdict(list)
    for var in ['itemId','title','price','URL','imageURL','imagefile','description',\
                'endtime','location','biketype']:
        bike_attrs[var] = []

    api = Shopping(config_file='ebay.yaml')

    if not(os.path.exists('data/'+'ebay_postings.csv')):
        fid = open('data/'+'ebay_postings.csv','w')
        for key in bike_attrs.keys():
            fid.write(key+',')
        fid.write('\n')
        fid.close()


    # Get unique counter for saving images in dir 'data'
    n = 0
    files = os.listdir('data')
    for name in files:
        if name.endswith('.jpg') and name.startswith('ebay'):
            if int(name[-9:-4]) > n:
                n = int(name[-9:-4]) + 1
    print("Starting at image",n)

    fid = open('data/'+'ebay_postings.csv','a')
    j = 0
    for posting_id in posting_ids:
        print("Posting",j,"of",n_items)
        response = api.execute('GetSingleItem', {'ItemID': posting_id,
                  'version': '981',
                  'IncludeSelector': ['PictureDetails','ItemSpecifics'],
                  'outputSelector': 'PictureURLLarge'})
        item = response.reply.Item
        
        try:
            bike_attrs['imageURL'].append(item.PictureURL[0])
            imagefile = 'ebay_image'+'{0:05d}'.format(n)+'.jpg'
            urllib.request.urlretrieve(item.PictureURL[0], 'data/'+imagefile)
            
            bike_attrs['imagefile'].append(imagefile)
            try:
                bike_attrs['location'].append(item.Location.replace(',',':'))
            except:
                locations.append([])
            bike_attrs['URL'].append(item.ViewItemURLForNaturalSearch)
            bike_attrs['itemId'].append(item.ItemID)
            
            bike_attrs['title'].append(item.Title.replace(',',' '))
            bike_attrs['price'].append(item.ConvertedCurrentPrice.value)
            bike_attrs['endtime'].append(str(item.EndTime.date()))
            bike_attrs['biketype'].append(get_biketype(item).replace(',',' '))

                    
            response = api.execute('GetSingleItem', {'ItemID': posting_id,
                  'version': '981',
                  'IncludeSelector': ['TextDescription']})
            item = response.reply.Item
            bike_attrs['description'].append(item.Description.replace('\n',' ').replace(',',' '))
            
            j += 1
            n += 1
            
            for key in bike_attrs.keys():
                fid.write(bike_attrs[key][-1]+', ')
            fid.write('\n')
            
        except:
            pass
            
    fid.close()

    return bike_attrs


def get_biketype(item):

    biketype = ''
    try:
        for line in item.ItemSpecifics.NameValueList:
            if line.get('Name') == 'Type':
                biketype = line.get('Value')
    except:
        try:
            if item.ItemSpecifics.NameValueList == 'Type':
                biketype = item.ItemSpecifics.NameValueList.Value
        except:
            pass
    print('biketype: ',biketype)

    return biketype   

posting_ids = get_ebay_postings('seattle')

new_posting_ids, deleted_posting_ids = check_against_saved_postings(posting_ids)

bike_attrs = get_new_posting_attrs(new_posting_ids)