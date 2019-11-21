import math
import numpy as np
import re
from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
import cv2 as cv
import os

from keras.utils import to_categorical
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend

import random


#file path
NEW_PATH = "./new_location/"
LOAD_FILE = 'Location_Reader2.h5' #trained model to load

#constants for training cnn
PATH = "/home/fizzer/enph353_cnn_lab/location_data/"
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4 
EPOCH = 30  #number of times data set is passed through the cnn
BATCH_SIZE = 16  #number of training examples
SAVE_FILE = 'Location_Reader2.h5' #file to save the trained model
CLASSES = 6


class ReadLocation:

  def __init__(self):
    self.conv_model = models.Sequential()

  # crop plates into 4 parts containing the individual characters
  def crop_and_label(self, img, filename, character_array):
    SIZE = 256, 128
    PLATE_SIZE = 600, 298
    BOX = (128,0,256,128)

    img = img.resize(SIZE)

    #img.save("/home/fizzer/enph353_cnn_lab/crops/" + 'location' + str(random.randint(0,999)) + '.png')

    im = img.crop(BOX)

    #im.save("/home/fizzer/enph353_cnn_lab/crops/" + str(random.randint(0,999)) + '.png')

    im = np.array(im)
    string = filename[1]
    pair = (im, string)
    character_array.append(pair)

    return


  # return an array of file names in a given folder 
  def files_in_folder(self, folder_path):

    self.files = os.listdir(folder_path)
    return self.files

  # loop through pictures and store in array of characters
  def return_character_array(self, path):
    images_array = self.files_in_folder(path)
    num_images = len(images_array)
    character_array = []
    
    for i in range (num_images):
      filename_i = images_array[i]
      self.crop_and_label(Image.open("{}{}".format(path + "/",filename_i)),"{}".format(filename_i), character_array)
    
    return character_array



  def get_encoder(self, data):
    data = array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    encoded = to_categorical(integer_encoded)
    invert_dict= dict(zip(integer_encoded, data))
    return encoded,invert_dict



  #splits data into character image (X) and corrisponding label (Y)
  def split_data(self, data):
    X = []
    Y_orig = []
    Y = []
    for e in data:
      X.append(e[0])
      Y_orig.append(e[1])
    Y,invert_dict = self.get_encoder(Y_orig)
    return X,Y,invert_dict


  #reset weights
  def reset_weights(self, model):
      session = backend.get_session()
      for layer in model.layers: 
          if hasattr(layer, 'kernel_initializer'):
              layer.kernel.initializer.run(session=session)


  # trains the cnn
  def train_nn(self, path):

    self.conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(128, 128, 3)))
    self.conv_model.add(layers.MaxPooling2D((2, 2)))
    self.conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    self.conv_model.add(layers.MaxPooling2D((2, 2)))
    self.conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    self.conv_model.add(layers.MaxPooling2D((2, 2)))
    self.conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    self.conv_model.add(layers.MaxPooling2D((2, 2)))
    self.conv_model.add(layers.Flatten())
    self.conv_model.add(layers.Dropout(0.5))
    self.conv_model.add(layers.Dense(512, activation='relu'))
    self.conv_model.add(layers.Dense(CLASSES, activation='softmax'))


    self.conv_model.summary()

    self.conv_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=LEARNING_RATE), metrics=['acc'])

    self.reset_weights(self.conv_model)

    #setup training dataset
    data = self.return_character_array(path)
    #print(data)
    X,Y,invert_dict = self.split_data(data)
    X = np.array(X)/255.0
    Y = np.array(Y)

    #train model
    history_conv = self.conv_model.fit(X, Y, validation_split=VALIDATION_SPLIT, epochs=EPOCH, batch_size=BATCH_SIZE)

    plt.plot(history_conv.history['loss'])
    plt.plot(history_conv.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()


    plt.plot(history_conv.history['acc'])
    plt.plot(history_conv.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
    plt.show()

    self.conv_model.save(SAVE_FILE)


  # predict new license plate
  def run_location_prediction(self, path):
    prediction = []
    files = self.files_in_folder(path)
    model = models.load_model(LOAD_FILE)
    #newest_file = files[0]

    data = self.return_character_array(path)
    X,Y,invert_dict = self.split_data(data)
    X = np.array(X)/255.0
    Y = np.array(Y)
    
    prediction = model.predict_classes(X)
    
    result = []

    for i in range(prediction.size):
      print('P' + str(prediction[i]))

    #print(prediction)
    #print(uncertainty)



def main():
  read_location = ReadLocation()

  # train cnn
  #read_location.train_nn(PATH)

  #predict new location
  read_location.run_location_prediction(NEW_PATH)


if __name__ == "__main__":    
    main() 