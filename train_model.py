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

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import os
from scipy import interp

# Data inputs
main_dir = '/Users/kehrl/Code/bike-finder/data/'
train_data_dir = main_dir+'/train/'
test_data_dir  = main_dir+'/test/'

# Dataset constants
n_classes = len(os.listdir(train_data_dir))

# Model inputs
model_name = 'vgg16'
batch_size = 10
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

    # Data augmentation for training set only
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0,
        zoom_range = 0.2,
        fill_mode = 'nearest'
        )
    
    # No data augmentation for the test set
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode = 'nearest'
        )

    train_generator = train_datagen.flow_from_directory(
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

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = image_shape,
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
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
    model.add(Dense(256, activation = 'relu', name = 'dense_11'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax'))

    opt = optimizers.RMSprop(lr=2e-4)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy',
                 metrics = ['accuracy'])

    checkpointer = ModelCheckpoint(filepath='model.best.hdf5', verbose=1, save_best_only=False)
   
    history = model.fit(train_data, train_y,
             epochs=epochs,
             batch_size=batch_size,
             validation_data = [test_data, test_y],
             callbacks = [checkpointer])
    
    # Save model results for predictions
    model.save_weights(top_model_weights_path)

    return model, history  

if __name__ == "__main__":

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

    model, history = train_top_model(train_data, train_y, test_data, test_y, n_classes, top_model_weights_path)

    # Make some plots
    # Loss over epochs
    #matplotlib.rc('font',family='sans-serif',size=22)
    plt.plot(history.history['val_loss'], lw = 2)
    plt.plot(history.history['loss'], lw = 2)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([0,100])
    plt.legend(['test', 'train'], loc='upper right')
    plt.subplots_adjust(top = 0.98, bottom = 0.15, left = 0.15, right = 0.94)
    plt.savefig('figures/loss_vs_epoch.png', format = 'PNG')
    plt.close()

    # ROC curve
    matplotlib.rc('font',size=22)
    # Compute macro-averaged ROC
    pred_y = model.predict(test_data)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds = roc_curve(test_y[:, i], pred_y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # First aggregate all false positive rates
    mean_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(mean_fpr)
    for i in range(n_classes):
        mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    auc_mean = auc(mean_fpr, mean_tpr)
    print("AUC is", auc_mean)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot(np.r_[0, mean_fpr], np.r_[0, mean_tpr], lw=3, color = 'b')

    plt.subplots_adjust(top = 0.98, bottom = 0.15, left = 0.15, right = 0.94)
    plt.ylabel('True positive rate'); plt.xlabel('False positive rate')
    plt.xlim([-0.01, 1]); plt.ylim([0,1.01])
    plt.savefig('figures/ROC.png', format = 'PNG')
    plt.close()
