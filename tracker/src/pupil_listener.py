#!/usr/bin/python
# -*- coding: UTF-8 -*-
from itertools import count
import zmq
from threading import Thread
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
from tracker.msg import GazePoint
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import datetime
import time

VERBOSE = False
SUB_URL = "tcp://127.0.0.1:1213" #tcp 주소지정
count=1

class zmq_sub(Thread):
    def __init__(self, topic):
        Thread.__init__(self)
        self.topic = "ROS_" + topic

    def run(self):
        global count
        print('Listener Start (topic : {})'.format(self.topic[4:]))
        subscriber = ctx.socket(zmq.SUB)
        subscriber.connect(SUB_URL)
        subscriber.setsockopt(zmq.SUBSCRIBE, self.topic)
        rate=rospy.Rate(30)
        while not rospy.is_shutdown():
            msg = subscriber.recv().split(':')
            if msg[2] == "None":
                pass
            elif msg[1] == 'gaze_point_3d':
                rate.sleep()
                gaze.gaze_point_3d.x = float(msg[4])
                gaze.gaze_point_3d.y = -float(msg[2])
                gaze.gaze_point_3d.z = -float(msg[3])

            elif msg[1] == 'norm_pos':
                gaze.norm_pos.x = float(msg[2])
                gaze.norm_pos.y = float(msg[3])

            elif msg[1] == 'gaze_normals_3d_right':
                gaze.gaze_normals_3d_right.x = float(msg[4])
                gaze.gaze_normals_3d_right.y = -float(msg[2])
                gaze.gaze_normals_3d_right.z = -float(msg[3])

            elif msg[1] == 'gaze_normals_3d_left':
                gaze.gaze_normals_3d_left.x = float(msg[4])
                gaze.gaze_normals_3d_left.y = -float(msg[2])
                gaze.gaze_normals_3d_left.z = -float(msg[3])

            elif msg[1] == 'eye_centers_3d_right':
                gaze.eye_centers_3d_right.x = float(msg[4])
                gaze.eye_centers_3d_right.y = -float(msg[2])
                gaze.eye_centers_3d_right.z = -float(msg[3])

            elif msg[1] == 'eye_centers_3d_left':
                gaze.eye_centers_3d_left.x = float(msg[4])
                gaze.eye_centers_3d_left.y = -float(msg[2])
                gaze.eye_centers_3d_left.z = -float(msg[3])

            elif msg[1] == 'calib_rotation':
                gaze.calib_rotation = float(msg[2])


            elif msg[1] == 'pupil_confidence':
                gaze.pupil_confidence = float(msg[2])

            elif msg[1] == 'label':
                gaze.label = int(msg[2])

                #rate.sleep()
                gaze_pub.publish(gaze)

            elif msg[1] == 'gray':

                bridge = CvBridge()
                stringData = ':'.join(e for e in msg[2:])
                data = np.fromstring(stringData, dtype='uint8')

                decimg=cv2.imdecode(data,1) 
                image_message = bridge.cv2_to_imgmsg(decimg, encoding="bgr8")
                decimg2=cv2.circle(decimg,(int(gaze.norm_pos.x*1280),int((1-gaze.norm_pos.y)*720)),15,(0,0,255),-1)
                count+=1                                                             
                img_pub.publish(image_message)




if __name__ == "__main__":

    try:
        rospy.init_node('pupil_listener', anonymous=True) #노드 생성
        gaze_pub = rospy.Publisher('gaze', GazePoint, queue_size=10) #gazepoint puplish        
        img_pub = rospy.Publisher('FOV', Image, queue_size=10) #img publish
        gaze = GazePoint()
        world_img = CompressedImage()
        world_img.format = "jpeg"        
        ctx = zmq.Context()
        sub_gaze = zmq_sub("gaze")
        sub_gaze.start()
        sub_frame = zmq_sub("frame")
        sub_frame.start()
    except rospy.ROSInterruptException:
        pass
