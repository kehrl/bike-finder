from scipy.spatial import distance
import sys
import numpy as np
import os

import keras
from keras.applications import InceptionV3
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator

def get_bottleneck_features(image_files, dir='data/'):
    print('Getting bottleneck features...')

    model = VGG16(weights="imagenet", include_top = False)
    image_shape = (224, 224)

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 0,
        width_shift_range = 0,
        height_shift_range = 0,
        shear_range = 0,
        zoom_range = 0,
        fill_mode = 'nearest'
        )

    images = np.zeros([len(image_files), image_shape[0], image_shape[0], 3]) 
    for i in range(len(image_files)):
        image = load_img(dir+image_files[i].strip(), target_size = image_shape)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        images[i] = image
    
    bottleneck_features = model.predict_generator(datagen.flow(images, shuffle = False))
            
    return bottleneck_features

def find_similar_bikes(input_image_file): 
    '''
    Find matches for the features of the selected bike, 
    according to cosine similarity.
    '''
    
    pred = intermediate_model_features(get_bottleneck_features([input_image_file]))
    
    if not(os.path.isfile('weights/bottleneck_features_test.npy')):
        sys.exit("Bottleneck features aren't saved.")
    else:
        bottleneck_features = np.load('weights/intermediate_bottleneck_features_craigslist.npy')
        image_files = np.load('weights/craigslist_images.npy')
    
    sims = np.zeros([bottleneck_features.shape[0], ])
    for i in range(bottleneck_features.shape[0]):
        sims[i] = distance.cosine(pred.flatten(), bottleneck_features[i].flatten())
    
    print('min sim = ' + str(np.max(sims)))
    
    return sims

def intermediate_model_features(bottleneck_features, top_model_path='model.best.hdf5'):

    model = keras.models.load_model(top_model_path)

    intermediate_layer_model = Model(inputs=model.input, \
        outputs=model.get_layer('dense_11').output)
        
    intermediate_features = intermediate_layer_model.predict(bottleneck_features)

    return intermediate_features
    
def predict_bike_type(image_file, dir='data', top_model_path='model.best.hdf5'):

    # Load top model
    model = keras.models.load_model(top_model_path)

    # Get bottleneck features
    bottleneck_features = get_bottleneck_features([image_file], dir=dir)
       
    preds = model.predict(bottleneck_features)
    print(preds)
    
    # Get bike label
    bike_labels=list(np.load("weights/train_labels.npy")[()])
    
    biketype = bike_labels[np.argmax(preds)]
    
    keras.backend.clear_session()

    print("Bike labeled as "+biketype)
    return biketype
