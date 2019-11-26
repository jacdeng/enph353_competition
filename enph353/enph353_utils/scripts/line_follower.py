import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
# from std_msgs.msg import Float64
import time
import sys
import cv2
import numpy as np

class LineFollower:

    def __init__(self):
        self.cam_path = '/R1/pi_camera/image_raw'
        self.vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.initial_move()
        self.listener = rospy.Subscriber(self.cam_path, Image, self.callback)
        self.data = None
        self.slice_num = 30
        self.frame = Image()
        self.bridge = CvBridge()
        
        self.watchout_for_crossing = True
        self.frame_count = 0
        self.scanning = False
        self.previous_crit = 0
        self.gogogo = 0 
        self.waitwaitwait = 0
        self.first_approach = True

        time.sleep(5)

    def initial_move(self):
        print("first moves")
        self.move("L")
        time.sleep(0.2)
        self.move("F")
        time.sleep(0.6)
        self.move("L_slow")
        time.sleep(0.5)
        self.move("stop")
        print("initial moves complete")

    def move_set(self, action):
        self.move("stop")
        print(action)
        if action == "L-90":
            self.move("L")
            time.sleep(0.5)
            self.move("stop")
        if action == "R-90":
            self.move("L")
            time.sleep(0.5)
            self.move("stop")
        if action == "crawl forward":
            self.move("F")
            time.sleep(0.1)
            self.move("stop")

    def move(self, action):
        vel_cmd = Twist()
        if action == "F":
            vel_cmd.linear.x = 0.2
            vel_cmd.linear.y = 0
            vel_cmd.linear.z = 0
            vel_cmd.angular.z = 0.0
        elif action == "L_slow":
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.05
        elif action == "L": 
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.05
        elif action == "L": 
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.1
        elif action == "R": 
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.1
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

        # Get image dimensions
        h = data.height
        w = data.width

        # run or nah
        run = True

        # do we try and detect the red line?
        if(self.watchout_for_crossing) and not (self.scanning):
            self.previous_crit = 0

            # recognizing the redline
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, (0,50,20), (5,255,255))
            mask2 = cv2.inRange(hsv, (175,50,20), (180,255,255))
            mask = cv2.bitwise_or(mask1, mask2)
            # mask has RED -> white, other colors -> black

            s = mask[h-3, 0:w]

            # there is white (red) and 
            if np.sum(s) > 60:
                self.watchout_for_crossing = False
                self.scanning = True
                run = False
                self.move("stop")

        elif not self.scanning:
            self.frame_count = self.frame_count+1
            if(self.frame_count == 50):
                self.frame_count = 0
                self.watchout_for_crossing = True
        
        if(self.scanning):
            run = False
            crit = self.find_criteria(h,w)
            print("------") 
            print(self.previous_crit)
            print(crit)
            print(int(self.previous_crit) - int(crit))

            if int(self.previous_crit) - int(crit) > -600000:
                time.sleep(0.1)

                if abs(int(self.previous_crit) - int(crit)) > 10000 and (self.first_approach):
                    self.waitwaitwait = self.waitwaitwait + 1
                    if(self.waitwaitwait > 5):
                        print("guy started crossing, first_approach=false")
                        self.first_approach = False
                        self.waitwaitwait = 0
                else:
                    self.waitwaitwait = 0

                if abs(int(self.previous_crit) - int(crit)) < 10000 and not (self.first_approach):
                    print("clear")
                    self.gogogo = self.gogogo + 1
                    if(self.gogogo > 2):
                        self.scanning = False
                        run = True
                        self.first_approach = True
                        self.gogogo = 0
                else:
                    self.gogogo = 0
                
            self.previous_crit = crit
            


        if(run) and not (self.scanning):
            # Turn image black and white and slice into thin images
            # Find which slice contains right curb
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            bw = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]
            state = [0]*self.slice_num
            state_num = w

            for i in range(self.slice_num): 
                
                s = bw[5*h/8:7*h/8, i*w/self.slice_num:(i+1)*w/self.slice_num]
                
                # there are white inside this slice
                # pref right side, so take the slice the right most.
                if np.sum(s) > 0:
                    state = [0]*self.slice_num
                    state[i] = 1
                    state_num = i

        # print(state_num)
        # cv2.imshow("debug", self.frame)
        # cv2.waitKey(10)

        if (run):
            self.follow(state_num)

    def find_criteria(self,h,w):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        bw = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]

        #finding the checking rectangle
        check_h = 6*h/8+60
        s = bw[check_h, :]
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

        shiftdown = 20
        size_inc = 30
        check = bw[5*h/8-size_inc+shiftdown :6*h/8+size_inc+shiftdown , l:r] 

        cv2.rectangle(self.frame, ( l, 5*h/8-size_inc+shiftdown ), ( r, 6*h/8+size_inc+shiftdown ), (0,255,0), 2)

        return np.sum(check)

    def follow(self, state):
        if state < 20:
            self.move("L")
        elif state > 28:
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