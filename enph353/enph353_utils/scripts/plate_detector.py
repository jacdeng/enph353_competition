import cv2 as cv
import numpy as np
import rospy
import sys
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import random


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

            #if contours fit correct dimensions, draw contour
            if self.contour_fit(max_center, cv_image):
                x, y, w, h = cv.boundingRect(max_center)
                plate_center = plate_center.append(max_center)

                cv.drawContours(cv_image, plate_center, 0, (0,255,0), 3)

                #crop plate
                license_plate = cv_image[y:y+h , x:x+w]

                cv.imshow('plate_number', license_plate)
                
                #save license plate
                cv.imwrite('/home/fizzer/enph353_cnn_lab/pictures' + str(random.randint(0,999)) + '.png', license_plate)

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

        width = w > 50 and w < 200
        height = h > 50 and h < 200
        area = cv.contourArea(center) > 7000

        if width and height and area:
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