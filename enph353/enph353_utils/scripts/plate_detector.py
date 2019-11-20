import cv2 as cv
import numpy as np
import rospy
import sys
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import random


TOP_TO_EXCLUDE = 0.18
BOT_TO_INCLUDE = 0.09

class DetectPlate:

    def __init__(self):
        self.image_sub = rospy.Subscriber('R1/pi_camera/image_raw', Image, self.callback)
        self.CvBridge = CvBridge()


    def callback(self,data):
        try:
            #convert ROS image to cv image
            cv_image = self.CvBridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        plate = self.get_contour(cv_image)

    # finds
    def get_contour(self, cv_image):
        res, mask = self.mask_image(cv_image)

        plate_center = []

        #get contours of the masked image
        image , contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 

        if len(contours) != 0:

            #find contour with the biggest area
            max_center = max(contours, key=cv.contourArea)
            #cv_image = cv.circle(cv_image, (int(max_center[0][0][0]),int(max_center[0][0][1])), 5, (0, 0, 255) , -1)

            #if contours fit correct dimensions, draw contour
            if self.contour_fit(max_center, cv_image):
                x, y, w, h = cv.boundingRect(max_center)
                plate_center.append(max_center)

                # cv.drawContours(cv_image, plate_center, -1, (0,255,0), 1)

                y_license_plate = int(BOT_TO_INCLUDE*y)
                y_QR = int(TOP_TO_EXCLUDE*y)

                # license plate
                cv.rectangle(cv_image, (x,y+h), (x+w,y+h+y_license_plate), (0,0,255))
                license_plate = cv_image[y+h:y+h+y_license_plate, x:x+w]
                
                # spot num
                cv.rectangle(cv_image, (x,y+y_QR), (x+w,y+h), (255,0,0))
                spot_num = cv_image[y+y_QR:y+h, x:x+w]

                #crop plate
                cv.imshow('plate_number', license_plate)
                cv.imshow('spot number', spot_num)
                
                print("CAUGHT IT")
                #save license plate
                # cv.imwrite('/home/fizzer/353_pics/license/pictures' + str(random.randint(0,999)) + '.png', license_plate)
                # cv.imwrite('/home/fizzer/353_pics/spot_num/pictures' + str(random.randint(0,999)) + '.png', spot_num)

        cv.imshow('color', cv_image)
        cv.waitKey(5) 
        return plate_center

        #masks given image into one that only contains the characters on license plate
    def mask_image(self, cv_image):
        # Convert BGR to HSV
        hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
            
        #define range of colour in HSV
        lower_blue = np.array([0,0,90]) #darker blue
        upper_blue = np.array([0,0,255]) #lighter blue

        #Threshold HSV image to only contain blue colours in range
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv.bitwise_and(cv_image,cv_image, mask= mask)

        return res, mask


    # checks if contours are within required dimensions
    def contour_fit(self, center, img):

        x, y, w, h = cv.boundingRect(center)

        width = w > 60 and w < 180
        height = h > 60 and h < 180
        area = cv.contourArea(center) > 9000
        aspect_ratio = float(h) / w < 1

        if width and height and area and aspect_ratio:
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