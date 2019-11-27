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

from random import randint

# define constants

PLATE_DATA = "/home/fizzer/enph353_cnn_lab/pictures/"
LOCATION_DATA = "/home/fizzer/enph353_cnn_lab/location_data/"

PLATE_NUM_SAVE_FILE = 'Plate_num.h5'
PLATE_LET_SAVE_FILE = 'Plate_let.h5'
LOCATION_NUM_SAVE_FILE = 'Location_num.h5'

VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4 
EPOCH = 100  #number of times data set is passed through the cnn
BATCH_SIZE = 16  #number of training examples
PLATE_LET_CLASSES = 26 # A to Z
PLATE_NUM_CLASSES = 10 # 0 to 10
LOCATION_NUM_CLASSES = 6 # 1 to 6

class TrainCNN:

    def __init__(self):
        self.conv_model = models.Sequential()
        
    
    # trains the cnn
    def train_nn(self, num_classes, save_file, data):

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
        self.conv_model.add(layers.Dense(num_classes, activation='softmax'))

        self.conv_model.summary()

        self.conv_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=LEARNING_RATE), metrics=['acc'])

        self.reset_weights(self.conv_model)

        #setup training dataset
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

        self.conv_model.save(save_file)

    def get_encoder(self, data):
        data = array(data)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data)
        encoded = to_categorical(integer_encoded)
        invert_dict= dict(zip(integer_encoded, data))
        return encoded,invert_dict


    # splits data into character image (X) and corrisponding label (Y)
    def split_data(self, data):
        X = []
        Y_orig = []
        Y = []
        for e in data:
            X.append(e[0])
            Y_orig.append(e[1])
        Y,invert_dict = self.get_encoder(Y_orig)
        return X,Y,invert_dict


    # reset weights
    def reset_weights(self, model):
        session = backend.get_session()
        for layer in model.layers: 
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)


    # return an array of file names in a given folder 
    def files_in_folder(self, folder_path):
        files = os.listdir(folder_path)
        return files

    # prepares the plate data
    def prep_data(self,path,plate):

        images_array = self.files_in_folder(path)
        num_images = len(images_array)
        num_data = []
        let_data = []
        
        for i in range (num_images):
            cv_image = cv.imread(path + images_array[i])
            filename_i = images_array[i]

            if plate:
                self.crop_n_label_plates(cv_image, filename_i, num_data, let_data)

            else:
                self.crop_n_label_locations(cv_image, filename_i, num_data)

        return num_data, let_data


    # crop plates into 4 parts containing the individual characters and label them
    def crop_n_label_plates(self, cv_image, filename, num_data, let_data):
        OFFSET = 50
        LETTER_WIDTH = 100 
        right = OFFSET
        DIM = 128, 128
        PLATE_SIZE = 600, 298

        resized = cv.resize(cv_image, PLATE_SIZE, interpolation = cv.INTER_AREA)

        for i in range (4):
            if i == 2: 
                left = right + LETTER_WIDTH
            else: 
                left = right 
            right = left + LETTER_WIDTH

            crop = resized[0:298, left:right]
            img = cv.resize(crop, DIM, interpolation = cv.INTER_AREA)
            

            #if i < 2:
            #    cv.imwrite('/home/fizzer/enph353_cnn_lab/PLcrop/' + str(randint(0,9999)) + '.png', img)
            #else:
            #    cv.imwrite('/home/fizzer/enph353_cnn_lab/PNcrop/' + str(randint(0,9999)) + '.png', img)


            img = np.array(img)
            string = filename[i+1]
            pair = (img, string)
            
            if i < 2:
                let_data.append(pair)
            else:
                num_data.append(pair)

        return

    
    # crop locations into individual numbers and label them
    def crop_n_label_locations(self, cv_image, filename, num_data):
        DIM = 256, 128

        resized = cv.resize(cv_image, DIM, interpolation = cv.INTER_AREA)
        crop = resized[0:128, 128:256]

        #cv.imwrite('/home/fizzer/enph353_cnn_lab/LNcrop/' + str(randint(0,9999)) + '.png', crop)

        img = np.array(crop)
        string = filename[1]
        pair = (img, string)
        num_data.append(pair)

        return



def main():
    train_cnn = TrainCNN()

    #  can only train 1 cnn at a time

    num_data, let_data = train_cnn.prep_data(PLATE_DATA, plate = True)
    #  train number model for plate reading
    #train_cnn.train_nn(PLATE_NUM_CLASSES, PLATE_NUM_SAVE_FILE, num_data)

    #  train letter model for plate reading
    #train_cnn.train_nn(PLATE_LET_CLASSES, PLATE_LET_SAVE_FILE, let_data)

    #  train model for location reading
    data,_ = train_cnn.prep_data(LOCATION_DATA, plate = False) 
    train_cnn.train_nn(LOCATION_NUM_CLASSES, LOCATION_NUM_SAVE_FILE, data)



if __name__ == "__main__":    
    main() 