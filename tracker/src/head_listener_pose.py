#!/usr/bin/env python
import numpy as np
import rospy
import math
import argparse
import cv2
import imutils
import time

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PolygonStamped
from threading import Thread
import zmq


SUB_URL = "tcp://127.0.0.1:1234"

class point_pub:
    def __init__(self, name, x, y, z):
        self.pub = rospy.Publisher(name, PointStamped, queue_size=10)
        self.data = Point()
        self.data.x, self.data.y, self.data.z = x, y, z
        self.pointstamp = PointStamped()
        self.pointstamp.point = self.data
        self.pointstamp.header.frame_id = "/camera_link"
        self.pub.publish(self.pointstamp)

class point_polygon:
    def __init__(self,name,x,y,z):
        self.pub = rospy.Publisher(name, PolygonStamped, queue_size=10)
        self.data = PolygonStamped()
        self.data.polygon.points.append(x)
        self.data.polygon.points.append(y)
        self.data.polygon.points.append(z)
        self.data.header.frame_id = "/camera_link"
        self.pub.publish(self.data)

class find_circle:

    def __init__(self,lower,upper):
        self.lower = lower #lower color
        self.upper = upper #upper color
        self.x, self.y, self.z = 0, 0, 0 # image xyz
        self.point_mat = np.array([[self.x],[self.y],[self.z]])
        self.X, self.Y, self.Z, = 0, 0, 0 # point xyz
        self.pre_X, self.pre_Y, self.pre_Z = 0, 0, 0 # previous point xyz

    def circle(self,img,color):
        bridge = CvBridge()
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, self.lower, self.upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            i = max(cnts, key=cv2.contourArea)
            M = cv2.moments(i)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            ((x, y), radius) = cv2.minEnclosingCircle(i)
            if radius > 10:
                cv2.circle(img, (int(x), int(y)), int(radius),color, -1) # x y
                cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), -1)
                self.x, self.y = int(x), int(y)

    def projection(self):
        self.point_mat = np.array([[self.x*self.z],[self.y*self.z],[self.z]])
        p_mat = np.array([[0,0,1],[-0.00160981,0,0.5177980],[0,-0.00160981,0.390396]])
        mat = np.dot(p_mat,self.point_mat)
        if (mat[0][0] == 0)and(mat[1][0] == 0)and(mat[2][0] == 0):
            self.X, self.Y, self.Z = self.pre_X, self.pre_Y, self.pre_Z
        else:
            self.X, self.Y, self.Z = mat[0][0], mat[1][0], mat[2][0]
            self.pre_X, self.pre_Y, self.pre_Z = mat[0][0], mat[1][0], mat[2][0]


def callback_depth(depth_image):
    global yellow1, yellow2, yellow3 	
    bridge = CvBridge()
    img_depth = bridge.imgmsg_to_cv2(depth_image)

    yellow1.z = img_depth[yellow1.y][yellow1.x]
    yellow1.projection()
    yellow2.z = img_depth[yellow2.y][yellow2.x]
    yellow2.projection()
    yellow3.z = img_depth[yellow3.y][yellow3.x]
    yellow3.projection()

    first = point_pub('first',yellow1.X*0.001,yellow1.Y*0.001,yellow1.Z*0.001)
    second = point_pub('second',yellow2.X*0.001,yellow2.Y*0.001,yellow2.Z*0.001)
    third = point_pub('third',yellow3.X*0.001,yellow3.Y*0.001,yellow3.Z*0.001)
    tri = point_polygon('tri',first.data,second.data, third.data)



def callback2(image):
    global yellow1, yellow2, yellow3
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(image,"bgr8")
    img = cv2.circle(img, (320,240), 5, (0,0,255), 2) # x y

    yellow1.circle(img,(0, 255, 0))
    yellow2.circle(img,(0, 255, 0))
    yellow3.circle(img,(0, 255, 0))

    image_message = bridge.cv2_to_imgmsg(img, encoding="bgr8")
    pub.publish(image_message)

class zmq_sub(Thread):

    def __init__(self, topic):
        Thread.__init__(self)
        self.topic = "ROS_" + topic
        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        print("init")

    def run(self):
        print('Listener Start (topic : {})'.format(self.topic[4:]))
        subscriber = ctx.socket(zmq.SUB)
        subscriber.connect(SUB_URL)
        subscriber.setsockopt(zmq.SUBSCRIBE, self.topic)

	while not rospy.is_shutdown():	        
            msg = subscriber.recv().split(':')
            if msg[1] == "None":
                print(1)
            else :
                self.yaw = msg[1]
                #self.pitch = msg[2]
                #self.roll = msg[3]
                print(2)
                print("yaw,pitch,roll : ", self.yaw)

#global class
#yellow 25-40, 60-255, 30-255
color_lower = (125, 40, 40)
color_high = (180, 255, 255)


yellow1 = find_circle(color_lower,color_high)
yellow2 = find_circle(color_lower,color_high)
yellow3 = find_circle(color_lower,color_high)
p_mat = np.array([[0,0,1],[-0.00160981,0,0.5177980],[0,-0.00160981,0.390396]])


if __name__ == "__main__":
    rospy.init_node('head_listener',anonymous=True)

    ctx = zmq.Context()
    sub_head = zmq_sub("head_pose")
    sub_head.start()
    
    pub = rospy.Publisher('test_img', Image, queue_size=10)
    sub = rospy.Subscriber('/camera/color/image_raw', Image, callback2)
    sub2 = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, callback_depth)


    


    rospy.spin()
