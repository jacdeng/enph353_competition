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
PLATE_NUM_LOAD_FILE = 'Plate_num.h5'
PLATE_LET_LOAD_FILE = 'Plate_let.h5'
LOCATION_NUM_LOAD_FILE = 'Location_num.h5'


class ReadInfo:
    def __init__(self):
        pass

    # predict license plate
    def run_plate_prediction(self, cv_image):
        prediction = []
        #print("trying to load cnn")

        num_model = models.load_model(PLATE_NUM_LOAD_FILE)
        let_model = models.load_model(PLATE_LET_LOAD_FILE)

        #print("loaded cnn")

        num_data, let_data = self.crop_plate(cv_image)
        num_data = np.array(num_data)/255.0
        let_data = np.array(let_data)/255.0
        
        let_prediction = let_model.predict_classes(let_data)
        num_prediction = num_model.predict_classes(num_data)

        for i in range(2):
            prediction.append(self.num_to_char(let_prediction[i]))
        for i in range(2):
            prediction.append(num_prediction[i])

        #print("predicted")

        return str(prediction)

    # predict location number
    def run_location_prediction(self, cv_image):
        model = models.load_model(LOCATION_NUM_LOAD_FILE)

        X = self.crop_location(cv_image)
        X = np.array(X)/255.0
        
        prediction = model.predict_classes(X)
        
        return (str(prediction + 1))


    # crops plate number 4 individual characters and converts it to an array
    def crop_plate(self, cv_image):
        OFFSET = 50
        LETTER_WIDTH = 100 
        right = OFFSET
        DIM = 128, 128
        PLATE_SIZE = 600, 298
        num_data = []
        let_data = []

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

            # cv.imshow('cropped', img)
            # cv.waitKey(0)

            if i < 2:
                let_data.append(img)
            else:
                num_data.append(img)

        return num_data, let_data

    
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
        # cv.imshow('cropped', crop)
        # cv.waitKey(0)

        return X


    # converts a number to its corrisponding letter
    def num_to_char(self, num):
        char = num
        for i in range(26):
            if num == i:
                char = chr(i + 65)

        return char



def main():
    read_info = ReadInfo()

    # predict new location
    location = cv.imread('/home/fizzer/enph353_cnn_lab/testdata/pictures411.png')
    location_result = read_info.run_location_prediction(location)
    print('prediction: P' + location_result)

    # predict new license plate
    plate = cv.imread('/home/fizzer/enph353_cnn_lab/testdata/pictures362.png')
    plate_result = read_info.run_plate_prediction(plate)
    print('prediction: ' + plate_result)



if __name__ == "__main__":    
        main() 