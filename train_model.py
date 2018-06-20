#!/usr/bin/env python

# ImageNet classifications:
# - bicycle-built-for-two, tandem bicycle, tandem
# - mountain bike, all-terrain bike, off-roader
# - ordinary, ordinary bicycle
# - push-bike
# - safety bicycle, safety bike
# - velocipede

from keras.applications import InceptionV3
from keras.applications import VGG16

from keras.models import Sequential

from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import numpy as np
import os

# Data inputs
main_dir = '/Users/kehrl/Code/bike-finder/data/'
train_data_dir = main_dir+'/train/'
test_data_dir  = main_dir+'/test/'

# Dataset constants
n_classes = len(os.listdir(train_data_dir))

# Model inputs
model_name = 'vgg16'
batch_size = 50
epochs = 50
top_model_weights_path = 'weights/my_model'

# Dictionary of models
model_options = {
	"vgg16": VGG16,
	"inception": InceptionV3,
}

if model_name == 'inception':
    image_shape = (299, 299) # InceptionV3
elif model_name == 'vgg16':
    image_shape = (224, 224) # VGG16

def save_bottleneck_features(network, image_shape, batch_size, test_data_dir, train_data_dir):
    ''' 
    Saves bottleneck features for testing and training.
    '''
    print('\n Saving Bottleneck Features...')


    model = network(weights="imagenet", include_top = False)

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 10,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0,
        zoom_range = 0.2,
        fill_mode = 'nearest'
        )

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size = image_shape,
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
        )

    if not(os.path.isfile('weights/bottleneck_features_train.npy')):
        train_data = model.predict_generator(train_generator, verbose=1)
        np.save(open('weights/bottleneck_features_train.npy','wb'), train_data)
    else:
        train_data = np.load('weights/bottleneck_features_train.npy')

    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size = image_shape,
        batch_size = batch_size,
        class_mode = None,
        shuffle = False,
        )
            
    if not(os.path.isfile('weights/bottleneck_features_test.npy')):
        test_data = model.predict_generator(test_generator, verbose=1)
        np.save(open('weights/bottleneck_features_test.npy','wb'), test_data)
    else:
        test_data = np.load('weights/bottleneck_features_test.npy')
        
    train_y = to_categorical(train_generator.classes)
    test_y = to_categorical(test_generator.classes)
    train_labels = train_generator.class_indices
    np.save(open('weights/train_y.npy','wb'), train_y)
    np.save(open('weights/test_y.npy','wb'), test_y)
    np.save(open('weights/train_labels.npy','wb'), train_labels)
        
    return train_data, test_data, train_y, train_labels, test_y

def train_top_model(train_data, train_y, test_data, test_y, n_classes, top_model_weights_path):
    ''' 
    Train top layer with bottleneck features as input
    '''

    print('\n Training the FC Layers...')
   
    model = Sequential()
    model.add(Flatten(input_shape = train_data.shape[1:]))
    model.add(Dense(2056, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1028, activation = 'relu'))
    model.add(Dense(n_classes, activation = 'softmax'))

    opt = optimizers.SGD(lr = 1.0e-4, momentum=0.9)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy',
                 metrics = ['accuracy'])

    checkpointer = ModelCheckpoint(filepath='model.best.hdf5', verbose=1, save_best_only=False)
   
    model.fit(train_data, train_y,
             epochs=epochs,
             batch_size=batch_size,
             validation_data = [test_data, test_y],
             callbacks = [checkpointer])

    model.save_weights(top_model_weights_path)

    return model  

# create bottleneck features if they don't already exist.
if (not os.path.isfile('weights/bottleneck_features_train.npy')) or (not os.path.isfile('weights/bottleneck_features_test.npy')):
    train_data, test_data, train_y, train_labels, test_y = \
        save_bottleneck_features(model_options[model_name], image_shape, \
        batch_size, test_data_dir, train_data_dir)
else:
    train_data = np.load('weights/bottleneck_features_train.npy')
    test_data = np.load('weights/bottleneck_features_test.npy')
    train_y = np.load('weights/train_y.npy')
    test_y = np.load('weights/test_y.npy') 
    train_labels = np.load('weights/train_labels.npy')

model = train_top_model(train_data, train_y, test_data, test_y, n_classes, top_model_weights_path)

#preds = model.predict(image)
#P = imagenet_utils.decode_predictions(preds)
