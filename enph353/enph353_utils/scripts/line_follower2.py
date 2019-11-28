import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
# from std_msgs.msg import Float64
import time
import sys
import cv2
import numpy as np
import plate_detector2

class LineFollower:

    def __init__(self):
        self.cam_path = '/R1/pi_camera/image_raw'
        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.listener = rospy.Subscriber(self.cam_path, Image, self.callback)
        self.data = None
        self.frame = Image()
        self.bridge = CvBridge()
        self.plate_detector = plate_detector2.DetectPlate()

        # time factor compensator, default 1 at realtimefactor = 1.6
        # no longer using this anymore so setting it to 0, use rospy.sleep instead
        self.factor = 0

        # frame counters
        self.frame_count_running = 0
        self.frame_count_checker = 0

        # starting state
        self.starting = True

        # hard left turning state
        self.left_turning = False
        
        # running state (does not check for crossing)
        self.running = False
        self.do_not_scan = False
        self.slice_amt = 40

        # scanning state
        self.scanning = False
        self.previous_crit = 0
        self.go_counter = 0 
        self.wait_counter = 0
        self.first_approach = True

    def move_set(self, action):
        if action == "crawl forward cnn":
            self.move("F")
            rospy.sleep(0.3)
            self.move("stop")
        if action == "crawl forward crosswalk":
            self.move("F")
            rospy.sleep(0.3)
            self.move("stop")

    def move(self, action):
        vel_cmd = Twist()
        if action == "F":
            vel_cmd.linear.x = 0.2
            vel_cmd.linear.y = 0
            vel_cmd.linear.z = 0
            vel_cmd.angular.z = 0.0
        elif action == "L": 
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.05
        elif action == "R": 
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.05
        elif action == "stop":
            vel_cmd.linear.x = 0
            vel_cmd.linear.y = 0
            vel_cmd.linear.z = 0
            vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)

    def callback(self, data):
        try:    
            self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        plates_found = False
        # refer to plate_detector2. check frame for potential license plates
        if self.frame_count_checker > 6 and not (self.scanning):
            plates_found = self.plate_detector.check_frame(self.frame)
            self.frame_count_checker = 0
            if plates_found:
                self.move_set("crawl forward cnn")
                spot_list = []
                license_list = []
                spot_list, license_list = self.plate_detector.get_info()
                if len(spot_list) != 0 and len(license_list) != 0:
                    self.plate_detector.publish()
                    pass
                else:
                    print("confidence low. not publishing")
        self.frame_count_checker = self.frame_count_checker+1

        # converting image into black and white from gray
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        bw = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]

        # Get image dimensions
        h = data.height
        w = data.width

        # Starting state --->
        if(self.starting) and not (self.left_turning) and not (self.scanning) and not (self.running):
            print(" initial state, moving forward ")
            self.move("F")
            if(self.has_white_at_center(bw,h,w, 7*h/8 + 30)):
                self.move("stop")
                self.starting = False
                self.left_turning = True

        # Left Turning state --->
        if(self.left_turning) and not (self.starting) and not (self.scanning) and not (self.running):
            print(" initial state, left turning ")
            self.move("L")
            if(self.has_white_at_right(bw,h,w)):
                self.move("stop")
                self.left_turning = False
                self.running = True

        # Line following state (running state) --->
        if(self.running) and not (self.starting) and not (self.scanning) and not (self.left_turning):

            self.frame_count_running = self.frame_count_running+1

            state = [0]*self.slice_amt
            target_state = w
            for i in range(self.slice_amt): 
                s = bw[5*h/8:7*h/8, i*w/self.slice_amt:(i+1)*w/self.slice_amt]
                # pref right side, so take the slice the right most.
                if np.sum(s) > 0:
                    state = [0]*self.slice_amt
                    state[i] = 1
                    target_state = i
            
            self.follow(target_state)

            # Trigger checker for crossings, only scan for crossing when frame_count_running is larger than 50
            if self.frame_count_running > 50:
                if self.has_red_line(h,w):
                    self.running = False
                    self.scanning = True
                    self.move_set("crawl forward crosswalk")
        
        # Scannning state (looking for pedestrians) --->
        if (self.scanning) and not (self.starting) and not (self.running) and not (self.left_turning):

            self.frame_count_running = 0

            crit = self.find_criteria(bw,h,w, 6*h/8)
            diff = int(self.previous_crit) - int(crit)
            print("------") 
            print(self.previous_crit)
            print(crit)
            print(diff)

            if abs(diff) < 550000  :

                # wait till a huge change happens then start checking for clears so our car can go
                if abs(diff) > 11000 and (self.first_approach):
                    print "diff over 10000"
                    self.wait_counter = self.wait_counter + 1
                    if(self.wait_counter > 6):
                        print("guy started crossing, first_approach=false")
                        self.first_approach = False
                        self.wait_counter = 0
                else:
                    if self.wait_counter>0:
                        self.wait_counter = self.wait_counter-1
                    else:
                        self.wait_counter = 0
                
                # wait till a sequence of small change happens meaning we have a clear, then we go
                if abs(diff) < 6000 and not (self.first_approach):
                    print("clear")
                    self.go_counter = self.go_counter + 1
                    if(self.go_counter > 4):
                        print("===================")
                        self.first_approach = True
                        self.go_counter = 0
                        print("--- SCAN COMPLETE ---")

                        self.running = True
                        self.scanning = False
                else:
                    self.go_counter = 0
            
            self.previous_crit = crit
        
        self.plate_detector.reset_spot_and_license()
        # cv2.imshow("debug", bw)
        # cv2.waitKey(10)

            
    def has_red_line(self,h,w):
        # recognizing the redline
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, (0,50,20), (5,255,255))
        mask2 = cv2.inRange(hsv, (175,50,20), (180,255,255))
        mask = cv2.bitwise_or(mask1, mask2)
        # mask has RED -> white, other colors -> black

        s = mask[h-3, 0:w]

        # there is white (red) and 
        if np.sum(s) > 60:
            return True
        else:
            return False


    def has_white_at_right(self,bw,h,w):

        # draw a small square in the right of the screen, if it is white, we start stop the left turn and start to line follow
        check_h = 6*h/8
        check_w = 6*w/8-113
        
        check_range = 7
        check = bw[check_h-check_range:check_h+check_range , check_w-check_range:check_w+check_range]
        cv2.rectangle(self.frame, ( check_w-check_range , check_h-check_range ), ( check_w+check_range , check_h+check_range), (0,255,0), 2)
        if np.sum(check)>5:
            return True
        else:
            return False

    def has_white_at_center(self,bw,h,w, scan_height):

        # draw a small square in the middle of the screen, if it is white, we start the left turn
        check_h = scan_height
        check_w = w/2
        check_range = 7
        check = bw[check_h-check_range:check_h+check_range , check_w-check_range:check_w+check_range]
        cv2.rectangle(self.frame, ( check_w-check_range , check_h-check_range ), ( check_w+check_range , check_h+check_range), (0,255,0), 2)
        if np.sum(check)>5:
            return True
        else:
            return False

    def find_criteria(self,bw,h,w, check_h):

        #finding the checking rectangle
        s = bw[check_h+60, :]
        l=0
        r=w
        for i in range(w):
            if(np.sum(bw[check_h,i]) > 0):
                l = i
                break
        
        for j in range(0,w):
            if(np.sum(bw[check_h,w-1-j]) > 0):
                r=w-j
                break
        
        height_of_scan_box = h/8
        shift = 20
        size_inc = 30
        check = bw[check_h-height_of_scan_box-size_inc+shift :check_h+size_inc+shift , l:r] 
        cv2.rectangle(bw, ( l, check_h-height_of_scan_box-size_inc+shift ), ( r, check_h+size_inc+shift ), (255,0,0), 2)

        return np.sum(check)

    def follow(self, state):
        if state < 31:
            self.move("L")
        elif state > 35:
            self.move("R")
        else:
            self.move("F")


def main(args):
    rospy.init_node('line_follower', anonymous=True)
    follower = LineFollower()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)