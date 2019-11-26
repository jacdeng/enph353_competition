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

# define constants
PLATE_LOAD_FILE = 'Plate_Reader5.h5'
LOCATION_LOAD_FILE = 'Location_Reader2.h5'


class ReadInfo:
    def __init__(self):
        pass

    # predict license plate
    def run_plate_prediction(self, cv_image):
        model = models.load_model(PLATE_LOAD_FILE)

        X = self.crop_plate(cv_image)
        X = np.array(X)/255.0
        
        prediction = model.predict_classes(X)
        result = []
        pre = True

        for i in range(prediction.size):
            if prediction[i] > 9 and i < 2:
                result.append(self.num_to_char(prediction[i]))
            elif prediction[i] < 9 and i > 1:
                result.append(prediction[i])
            else:
                if prediction[i] > 9:
                    result.append(self.num_to_char(prediction[i]))
                pre = False

        if pre != True:
            print("prediction is probably wrong")

        print(result)

    # predict location number
    def run_location_prediction(self, cv_image):
        model = models.load_model(LOCATION_LOAD_FILE)

        X = self.crop_location(cv_image)
        X = np.array(X)/255.0
        
        prediction = model.predict_classes(X)
        
        print('P' + str(prediction))


    # crops plate number 4 individual characters and converts it to an array
    def crop_plate(self, cv_image):
        OFFSET = 50
        LETTER_WIDTH = 100 
        right = OFFSET
        DIM = 128, 128
        PLATE_SIZE = 600, 298
        X = []

        resized = cv.resize(cv_image, PLATE_SIZE, interpolation = cv.INTER_AREA)

        cv.imshow('resized', resized)
        cv.waitKey(0)
        
        for i in range (4):
            if i == 2: 
                left = right + LETTER_WIDTH
            else: 
                left = right 
            right = left + LETTER_WIDTH

            crop = resized[0:298, left:right]
            img = cv.resize(crop, DIM, interpolation = cv.INTER_AREA)

            img = np.array(img)
            X.append(img)

            cv.imshow('cropped', img)
            cv.waitKey(0)

        return X

    
    # crops location into an image of the number and converts it to an array 
    def crop_location(self, cv_image):
        DIM = 256, 128
        X = []

        resized = cv.resize(cv_image, DIM, interpolation = cv.INTER_AREA)
        crop = resized[0:128, 128:256]

        img = np.array(crop)

        X.append(img)

        cv.imshow('resized', resized)
        cv.waitKey(0)
        cv.imshow('cropped', crop)
        cv.waitKey(0)

        return X


    # converts a number to its corrisponding letter
    def num_to_char(self, num):
        char = num
        for i in range(10 , 35):
            if num == i:
                char = chr(i + 55)

        return char



def main():
  read_info = ReadInfo()

  #predict new location
  location = cv.imread('/home/fizzer/enph353_cnn_lab/new_locations/pictures662.png')
  read_info.run_location_prediction(location)

  plate = cv.imread('/home/fizzer/enph353_cnn_lab/new_data/aZF86.png')
  read_info.run_plate_prediction(plate)



if __name__ == "__main__":    
    main() 