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
NEW_PLATE_PATH = "/home/fizzer/353_ws/src/enph353_competition/enph353/enph353_utils/scripts/new_plates/"
LOAD_FILE = 'Plate_Reader5.h5' #trained model to load


#constants for training cnn
PATH = "/home/fizzer/enph353_cnn_lab/pictures/"
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4 
EPOCH = 20  #number of times data set is passed through the cnn
BATCH_SIZE = 16  #number of training examples
SAVE_FILE = 'Plate_Reader5.h5' #file to save the trained model


class ReadPlate:

  def __init__(self):
    self.conv_model = models.Sequential()



  # crop plates into 4 parts containing the individual characters
  def crop_and_label(self, img, filename, character_array):
    OFFSET = 50
    LETTER_WIDTH = 100 
    right = OFFSET
    size = 128, 128
    PLATE_SIZE = 600, 298

    img = img.resize(PLATE_SIZE)

    #img.save("/home/fizzer/enph353_cnn_lab/crops/" + 'plate' + str(random.randint(0,999)) + '.png')

    for x in range (4):
      if x == 2: 
        left = right + LETTER_WIDTH
      else: 
        left = right 
      right = left + LETTER_WIDTH
      box=(left,0,right,298)
      im = img.crop(box)
      im = im.resize(size) #resize image

      #im.save("/home/fizzer/enph353_cnn_lab/crops/" + str(random.randint(0,999)) + '.png')

      im = np.array(im)
      string = filename[x+1]
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

  '''
  #prepares data to train nn
  def setup_data(self, path):

    #setup training dataset
    data = self.return_character_array(path)
    X,Y,invert_dict = self.split_data(data)
    X = np.array(X)/255.0
    Y = np.array(Y)

    return X, Y
  '''

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
    self.conv_model.add(layers.Dense(36, activation='softmax'))


    self.conv_model.summary()

    LEARNING_RATE = 1e-4
    self.conv_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=LEARNING_RATE), metrics=['acc'])

    self.reset_weights(self.conv_model)

    #setup training dataset
    data = self.return_character_array(path)
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
  def run_plate_prediction(self, path):
    prediction = []
    files = self.files_in_folder(path)
    model = models.load_model(LOAD_FILE)
    #newest_file = files[0]

    data = self.return_character_array(path)
    X,Y,invert_dict = self.split_data(data)
    X = np.array(X)/255.0
    Y = np.array(Y)
    
    prediction = model.predict_classes(X)

    print(prediction)

    #f = backend.function([model.layers[0].input, backend.learning_phase()], [model.layers[-1].output])
    
    #_,uncertainty = self.predict_with_uncertainty(f, X)
    
    result = []

    for i in range(prediction.size):
      #print(i)
      if prediction[i] > 9 and i < 2:
        result.append(self.num_to_char(prediction[i]))
      elif prediction[i] < 9 and i > 1:
        result.append(prediction[i])
      else:
        print("prediction is probably wrong")

    print(result)
    #print(uncertainty)

  def run_location_prediction(self, path):
    image_array = self.files_in_folder(path)
    num_images = len(image_array)
    num_array = []

    for i in range(num_images):
      filename_i = image_array[i]
      self.crop_location(Image.open("{}{}".format(path + "/",filename_i)),"{}".format(filename_i), num_array)
    

  def crop_location(self, img, filename, num_array):
    SIZE = 256, 128
    BOX = (128, 0, 256, 128)
    img.resize(SIZE)
    img.crop(BOX)
    img.save("/home/fizzer/enph353_cnn_lab/crops/" + 'plate' + str(random.randint(0,999)) + '.png')



  '''def predict_with_uncertainty(self, f, x, n_iter=10):
    result = np.zeros((n_iter,) + x.shape)

    for iter in range(n_iter):
        result[iter] = f(x, 1)

    prediction = result.mean(axis=0)
    uncertainty = result.var(axis=0)
    return prediction, uncertainty
    '''

  def num_to_char(self, num):
    char = num
    for i in range(10 , 35):
      if num == i:
        char = chr(i + 55)

    return char




def main():
  read_plate = ReadPlate()

  # train cnn
  #read_plate.train_nn(PATH)

  # predict new license plate
  read_plate.run_plate_prediction(NEW_PLATE_PATH)


if __name__ == "__main__":    
    main() 