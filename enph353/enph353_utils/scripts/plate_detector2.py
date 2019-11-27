import cv2 as cv
import numpy as np
import rospy
import sys
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import random
import read_info
from std_msgs.msg import String

# INTERFACE FOR PLATE_DETECTOR1 AND THE CNN FOR LINE_FOLLOWER2.PY
# refactored to work as subsidiary to line_follower2.py
# also does the publishing to rosnode, interacts with read_info

TOP_TO_EXCLUDE = 0.14
BOT_TO_INCLUDE = 0.07

TEAM_ID = str("teamRocket")
TEAM_PASSWORD = str("mountainDew")

class DetectPlate:

    def __init__(self):
        self.info_pub = rospy.Publisher('/license_plate', String, queue_size=1)
        self.info_pub.publish(TEAM_ID+","+TEAM_PASSWORD+"0,XX00")
        self.reader = read_info.ReadInfo()
        self.spot_num = None
        self.license_plate = None
        self.spot_num_list = []
        self.license_plate_list = []

    # find the very specific blue color
    # not using this anymore
    def get_that_blue(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # define range of blue color in HSV

        lower_blue = np.uint8([[[90,7,7 ]]])
        upper_blue = np.uint8([[[105,40,40 ]]])

        lower = cv.cvtColor(lower_blue,cv.COLOR_BGR2HSV)
        upper = cv.cvtColor(upper_blue,cv.COLOR_BGR2HSV)
        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, lower, upper)
        res = cv.bitwise_and(frame,frame, mask= mask)
        return res

    # uses red_info and gets the strings
    def get_info(self):
        pred1 = pred2 = False
        if self.spot_num is not None and self.license_plate is not None:
            pred1, self.license_plate_list = self.reader.run_plate_prediction(self.license_plate)
            if pred1:
                pred2, self.spot_num_list = self.reader.run_location_prediction(self.spot_num)

        print("+++++++++++++++++++++++++++++")
        print(self.spot_num_list)
        print(self.license_plate_list)
        print("+++++++++++++++++++++++++++++")

        if not pred1:
            self.license_plate_list = []

        if not pred2:
            self.spot_num_list = []

        return self.spot_num_list, self.license_plate_list

    # reset self.spot_num and self.license_plate
    def reset_spot_and_license(self):
        self.spot_num = None
        self.license_plate = None
        self.spot_num_list = []
        self.license_plate_list = []

    # publish to node
    def publish(self):
        print("trying to publishing")
        str_to_pub = ""
        if self.spot_num is not None and self.license_plate is not None:
            if self.spot_num_list is not [] and self.license_plate_list is not []:
                str_to_pub1 = TEAM_ID+","+TEAM_PASSWORD + ","

                str_to_pub2 = ""
                for i in self.spot_num_list:
                    str_to_pub2 = str_to_pub2 + str(i)
                
                str_to_pub3 = ""
                for j in range(len(self.license_plate_list)):
                    if j == 0 or j == 1:
                        str_to_pub3 = str_to_pub3 + str(self.license_plate_list[j])
                    if j == 2 or j == 3:
                        str_split = list(str(self.license_plate_list[j]))
                        str_to_pub3 = str_to_pub3 + str(str_split[1])
        
                str_to_pub = str_to_pub1 + str_to_pub2 + "," + str_to_pub3

                print("+++++++++++++++++++++++++++++")
                print(str_to_pub)
                print("+++++++++++++++++++++++++++++")
                self.info_pub.publish(str_to_pub)


    # finds the plate and spot num from the given frame
    # returns true if found a plate and spotnum, else returns false
    def check_frame(self, frame):
        flag = False
        res, mask = self.mask_image(frame)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        bw = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)[1]
        
        #dim_width = frame.shape[1]
        #dim_height = frame.shape[0]

        #get contours of the masked image
        image , contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 

        if len(contours) != 0:

            #find contour with the biggest area
            max_center = max(contours, key=cv.contourArea)
            #frame = cv.circle(frame, (int(max_center[0][0][0]),int(max_center[0][0][1])), 5, (0, 0, 255) , -1)
            #cv.drawContours(frame, max_center, -1, (0,255,0), 2)

            x, y, w, h = cv.boundingRect(max_center)
            #cv.circle(frame, (int(x),int(y)), 5, (0,0,255),1)

            #if contours fit correct dimensions, draw contour
            if self.contour_fit(max_center, frame):
                x, y, w, h = cv.boundingRect(max_center)

                # cv.drawContours(frame, plate_center, -1, (0,255,0), 1)

                y_license_plate = int(BOT_TO_INCLUDE*y)
                y_QR = int(TOP_TO_EXCLUDE*y)

                # license plate
                #cv.rectangle(frame, (x,y+h), (x+w,y+h+y_license_plate), (0,0,255))
                license_plate = frame[y+h:y+h+y_license_plate, x:x+w]
                
                # spot num
                #cv.rectangle(frame, (x,y+y_QR), (x+w,y+h), (255,0,0))
                spot_num = frame[y+y_QR:y+h, x:x+w]

                if self.is_valid_plate(license_plate) and self.is_valid_spot(spot_num):
                    
                    flag = True
                    # uncomment cv.imwrite to save the read files to folder

                    # cv.imshow('plate_number', license_plate)
                    # cv.waitKey(2)
                    cv.imwrite('./new_plates/' + str(random.randint(0,999)) + '.png', license_plate)
                    self.license_plate = license_plate
                
                    # cv.imshow('spot number', spot_num)
                    # cv.waitKey(2)
                    cv.imwrite('./new_location/' + str(random.randint(0,999)) + '.png', spot_num)
                    self.spot_num = spot_num

                #save license plate
        
        return flag

    def is_valid_plate(self, license_plate):
        ar = license_plate.shape[0]/license_plate.shape[1]
        gray = cv.cvtColor(license_plate, cv.COLOR_BGR2GRAY)
        bw = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)[1]
        if np.sum(bw) > 0 and ar<0.2:
            return False
        else:
            return True
    
    def is_valid_spot(self, spot_num):
        ar = spot_num.shape[0]/spot_num.shape[1]
        gray = cv.cvtColor(spot_num, cv.COLOR_BGR2GRAY)
        bw = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)[1]
        if np.sum(bw) > 0 and ar<0.5:
            return False
        else:
            return True

    def has_blue(self, blue):
        h = blue.shape[0]
        s = blue[5*h/8:7*h/8, :]
        if np.sum(s) > 0:
            return True
        else:
            return False

        #masks given image into one that only contains the characters on license plate
    def mask_image(self, cv_image):
        # Convert BGR to HSV
        hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
            
        #define range of colour in HSV
        lower_blue = np.array([0,0,90]) #darker blue
        upper_blue = np.array([0,0,224]) #lighter blue

        #Threshold HSV image to only contain blue colours in range
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv.bitwise_and(cv_image,cv_image, mask= mask)

        return res, mask

    # checks if contours are within required dimensions
    def contour_fit(self, center, img):

        x, y, w, h = cv.boundingRect(center)

        x_on_left = x<int(img.shape[1]/2)
        width = w > 40 and w < 210
        height = h > 40 and h < 210
        area = cv.contourArea(center) > 6000
        aspect_ratio = False

        if float(h) / w < 1 and 0.5 < float(h) / w:
            aspect_ratio = True

        on_page_req = x > 0

        if width and height and area and aspect_ratio and on_page_req and x_on_left:
            return True
        else:
            return False


def main(args):

    rospy.init_node('image_processor', anonymous=True)
    DetectPlate()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv.destroyAllWindows


if __name__ == '__main__':
    main(sys.argv)