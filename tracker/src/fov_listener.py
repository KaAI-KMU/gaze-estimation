#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

#from tkinter.messagebox import RETRY
#from unittest.mock import _CallValue
from tkinter import image_names
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from tracker.msg import gazepoint
import camera_models #shared_module/camera_models.py
import numpy as np
#import typing as T
from methods import normalize
from utils import _clamp_norm_point
import zmq
from threading import Thread
# Instantiate CvBridge
from sensor_msgs.msg import CompressedImage
import base64
global pose
pose = [0,0]

VERBOSE = False
SUB_URL = "tcp://127.0.0.1:3000"

V_EYE=1
count=1

v_eye_v=gazepoint()

def callback(msg):
    global pose

    vx=msg.v_eye.x
    vy=msg.v_eye.y
    vz=msg.v_eye.z
    #print(vz)

    intrinsics = camera_models.Camera_Model._from_raw_intrinsics(
    cam_name="scene camera",
    resolution=(1280, 720), #해상도
    intrinsics={
        "dist_coefs": #왜곡계수
        [
            [
            -0.3758628065070806,
            0.1643326166951343,
            0.00012182540692089567,
            0.00013422608638039466,
            0.03343691733865076,
            0.08235235770849726,
            -0.08225804883227375,
            0.14463365333602152,
            ]
        ],
        "camera_matrix": [ #camera intrinsic
        [794.3311439869655, 0.0, 633.0104437728625],
        [0.0, 793.5290139393004, 397.36927353414865],
        [0.0, 0.0, 1.0],
        ],
        "cam_type": "radial", #렌즈왜곡 중 방사왜곡
    }   
    )


    gaze_point_3d=[[vx,vy,vz]]
    #print(gaze_point_3d)


    image_point = intrinsics.projectPoints(
        np.array([gaze_point_3d])
    )

    image_point = image_point.reshape(-1, 2) 
    image_point = normalize(image_point[0], intrinsics.resolution, flip_y=True)

    image_point=_clamp_norm_point(image_point)
    pose[0] = image_point[0]
    pose[1] = image_point[1] #decimg=cv2.imdecode(imgz,cv2.IMREAD_COLOR)
    #print(pose[1])

def image_callback(msg):
    global pose, count
    print(1)
    norm_x=pose[0]
    norm_y=pose[1]
    print(norm_x,norm_y)
    #img=np.frombuffer(msg.data,dtype=np.uint8)
    imgz=np.fromstring(msg.data,dtype='uint8').reshape(msg.height, msg.width, -1)
    #imgz=np.frombuffer(msg.data,dtype='uint8')
    
    #print(imgz)
    #decimg=cv2.imdecode(imgz,cv2.IMREAD_COLOR)
    #decimg=base64.b64decode(imgz,cv2.IMREAD_COLOR)
    #print(decimg)
    #print(decimg)
    #print(norm_x)   
    decimg2=cv2.circle(imgz,(int(norm_x*1280),int(1-norm_y)*720),15,(0,0,255),-1)
    #decimg2=cv2.circle(imgz,(450,450),15,(0,0,255),-1)
    # #print(decimg2)
    cv2.imwrite("/home/kaai/pupil/recordings/imgdata2/%d.jpg" % count, decimg2)
                
    count += 1    

while not rospy.is_shutdown():
    rospy.init_node('gaze_listener')

    gaze_topic = "vector_v_eye"

    rospy.Subscriber(gaze_topic, gazepoint,callback)
    rospy.Subscriber("FOV",Image,image_callback)


    rospy.spin()
