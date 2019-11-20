import cv2 as cv
import numpy as np
import rospy
import sys
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import random


TOP_TO_EXCLUDE = 0.16
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

        #blue = self.get_that_blue(cv_image)

        plate = self.get_contour(cv_image)

    # find the very specific blue color
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


    # finds contours
    def get_contour(self, cv_image):
        res, mask = self.mask_image(cv_image)
        
        gray = cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)
        bw = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)[1]

        plate_center = []
        
        #dim_width = cv_image.shape[1]
        #dim_height = cv_image.shape[0]

        #get contours of the masked image
        image , contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 

        if len(contours) != 0:

            #find contour with the biggest area
            max_center = max(contours, key=cv.contourArea)
            #cv_image = cv.circle(cv_image, (int(max_center[0][0][0]),int(max_center[0][0][1])), 5, (0, 0, 255) , -1)
            cv.drawContours(cv_image, max_center, -1, (0,255,0), 2)

            x, y, w, h = cv.boundingRect(max_center)
            #cv.circle(cv_image, (int(x),int(y)), 5, (0,0,255),1)

            #if contours fit correct dimensions, draw contour
            if self.contour_fit(max_center, cv_image):
                x, y, w, h = cv.boundingRect(max_center)
                plate_center.append(max_center)

                # cv.drawContours(cv_image, plate_center, -1, (0,255,0), 1)

                y_license_plate = int(BOT_TO_INCLUDE*y)
                y_QR = int(TOP_TO_EXCLUDE*y)

                # license plate
                #cv.rectangle(cv_image, (x,y+h), (x+w,y+h+y_license_plate), (0,0,255))
                license_plate = cv_image[y+h:y+h+y_license_plate, x:x+w]
                
                # spot num
                #cv.rectangle(cv_image, (x,y+y_QR), (x+w,y+h), (255,0,0))
                spot_num = cv_image[y+y_QR:y+h, x:x+w]

                print("checking if valid")
                if self.is_valid_plate(license_plate) and self.is_valid_spot(spot_num):

                    print("ye")
                    cv.imshow('plate_number', license_plate)
                    cv.waitKey(2)
                    cv.imwrite('/home/fizzer/353_pics/license/pictures' + str(random.randint(0,999)) + '.png', license_plate)
                
                    cv.imshow('spot number', spot_num)
                    cv.waitKey(2)
                    cv.imwrite('/home/fizzer/353_pics/spot_num/pictures' + str(random.randint(0,999)) + '.png', spot_num)

                #save license plate

        cv.imshow('frame', cv_image)
        cv.waitKey(3) 
        return plate_center

    def is_valid_plate(self, license_plate):
        ar = license_plate.shape[0]/license_plate.shape[1]
        print(ar)
        gray = cv.cvtColor(license_plate, cv.COLOR_BGR2GRAY)
        bw = cv.threshold(gray, 225, 255, cv.THRESH_BINARY)[1]
        if np.sum(bw) > 0 and ar<0.2:
            return False
        else:
            return True
    
    def is_valid_spot(self, spot_num):
        ar = spot_num.shape[0]/spot_num.shape[1]
        print(ar)
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