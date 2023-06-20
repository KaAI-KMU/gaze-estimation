#!/usr/bin/env python

import rospy
import numpy as np
from tracker.msg import GazePoint
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

GAZE_POINT_3D = 1
NORM_POS = 1
FOV=1 
GAZE_NORMALS_3D_RIGHT=1
EYE_CENTERS_3D_RIGHT=1
GAZE_NORMALS_3D_LEFT=1
EYE_CENTERS_3D_LEFT=1
PUPIL_CONFIDENCE=1

bridge = CvBridge()

class LPF4pupil:

    def __init__(self, sens):

        self.sensitivity = sens
        self.sensorValue_3d = np.array([0, 0, 0])
        self.sensorValue_norm = np.array([0, 0])

        self.filterValue_3d = np.array([0, 0, 0])
        self.filterValue_norm = np.array([0, 0])


        self.filterValue_direction_3d_left = np.array([0, 0, 0])
        self.filterValue_eye_center_3d_left = np.array([0, 0,0])

        self.filterValue_direction_3d_right = np.array([0, 0, 0])
        self.filterValue_eye_center_3d_right = np.array([0, 0, 0])

        self.calib_rotation = 0.
        self.start = 0
        self.label = 0

        self.pupil_confidence=0. #cw

    def callback(self, msg):

        if GAZE_POINT_3D:
            self.sensorValue_3d = np.array([msg.gaze_point_3d.x, msg.gaze_point_3d.y, msg.gaze_point_3d.z])
        if NORM_POS:
            self.sensorValue_norm = np.array([msg.norm_pos.x, msg.norm_pos.y])

        if GAZE_NORMALS_3D_RIGHT:
            self.sensorValue_direction_3d_right = np.array([msg.gaze_normals_3d_right.x, msg.gaze_normals_3d_right.y, msg.gaze_normals_3d_right.z])
        if EYE_CENTERS_3D_RIGHT:
            self.sensorValue_eye_center_3d_right = np.array([msg.eye_centers_3d_right.x, msg.eye_centers_3d_right.y, msg.eye_centers_3d_right.z])

        if GAZE_NORMALS_3D_LEFT:
            self.sensorValue_direction_3d_left = np.array([msg.gaze_normals_3d_left.x, msg.gaze_normals_3d_left.y, msg.gaze_normals_3d_left.z])
        if EYE_CENTERS_3D_LEFT:
            self.sensorValue_eye_center_3d_left = np.array([msg.eye_centers_3d_left.x, msg.eye_centers_3d_left.y, msg.eye_centers_3d_left.z])

        self.pupil_confidence=np.array([msg.pupil_confidence])

        if self.start == 0:
            self.filterValue_3d = self.sensorValue_3d
            self.filterValue_norm = self.sensorValue_norm      
            self.filterValue_direction_3d_right = self.sensorValue_direction_3d_right
            self.filterValue_eye_center_3d_right = self.sensorValue_eye_center_3d_right

            self.filterValue_direction_3d_left = self.sensorValue_direction_3d_left
            self.filterValue_eye_center_3d_left = self.sensorValue_eye_center_3d_left

            self.start = 1
        if GAZE_POINT_3D:
            self.filterValue_3d = self.filterValue_3d * (1 - self.sensitivity) + self.sensorValue_3d * self.sensitivity
        if NORM_POS:
            self.filterValue_norm = self.filterValue_norm * (1 - self.sensitivity) + self.sensorValue_norm * self.sensitivity
        if GAZE_NORMALS_3D_LEFT:
            self.filterValue_direction_3d_left = self.filterValue_direction_3d_left * (1 - self.sensitivity) + self.sensorValue_direction_3d_left * self.sensitivity
        if EYE_CENTERS_3D_LEFT:
            self.filterValue_eye_center_3d_left = self.filterValue_eye_center_3d_left * (1 - self.sensitivity) + self.sensorValue_eye_center_3d_left  * self.sensitivity

        if GAZE_NORMALS_3D_RIGHT:
            self.filterValue_direction_3d_right = self.filterValue_direction_3d_right* (1 - self.sensitivity) + self.sensorValue_direction_3d_right * self.sensitivity
        if EYE_CENTERS_3D_RIGHT:
            self.filterValue_eye_center_3d_right = self.filterValue_eye_center_3d_right * (1 - self.sensitivity) + self.sensorValue_eye_center_3d_right * self.sensitivity

        self.label = msg.label
        self.pupil_confidence= self.pupil_confidence#cw

if __name__ == '__main__':
    lpf = LPF4pupil(0.6) #sensitivity = 0.6
    rospy.init_node('pupil_LPF', anonymous=True)

    pub1 = rospy.Publisher('gaze_LPF', GazePoint, queue_size=10)
    pub2 = rospy.Publisher('FOV_LPF', Image, queue_size=10) #cw

    rospy.Subscriber('gaze', GazePoint, lpf.callback)
    rospy.Subscriber('FOV', Image) #cw
    rate=rospy.Rate(30)
    while not rospy.is_shutdown():

        msg_filtered = GazePoint()
        msg_filtered2=Image()

        if GAZE_POINT_3D:
            msg_filtered.gaze_point_3d.x = lpf.filterValue_3d[0]
            msg_filtered.gaze_point_3d.y = lpf.filterValue_3d[1]
            msg_filtered.gaze_point_3d.z = lpf.filterValue_3d[2]
           
        if NORM_POS:
            msg_filtered.norm_pos.x = lpf.filterValue_norm[0]
            msg_filtered.norm_pos.y = lpf.filterValue_norm[1]

        if GAZE_NORMALS_3D_LEFT:
            msg_filtered.gaze_normals_3d_left.x = lpf.filterValue_direction_3d_left[0]
            msg_filtered.gaze_normals_3d_left.y = lpf.filterValue_direction_3d_left[1]
            msg_filtered.gaze_normals_3d_left.z = lpf.filterValue_direction_3d_left[2]

        if EYE_CENTERS_3D_LEFT:
            msg_filtered.eye_centers_3d_left.x = lpf.filterValue_eye_center_3d_left[0]
            msg_filtered.eye_centers_3d_left.y = lpf.filterValue_eye_center_3d_left[1]
            msg_filtered.eye_centers_3d_left.z = lpf.filterValue_eye_center_3d_left[2]


        if GAZE_NORMALS_3D_RIGHT:
            msg_filtered.gaze_normals_3d_right.x = lpf.filterValue_direction_3d_right[0]
            msg_filtered.gaze_normals_3d_right.y = lpf.filterValue_direction_3d_right[1]
            msg_filtered.gaze_normals_3d_right.z = lpf.filterValue_direction_3d_right[2]

        if EYE_CENTERS_3D_RIGHT:
            msg_filtered.eye_centers_3d_right.x = lpf.filterValue_eye_center_3d_right[0]
            msg_filtered.eye_centers_3d_right.y = lpf.filterValue_eye_center_3d_right[1]
            msg_filtered.eye_centers_3d_right.z = lpf.filterValue_eye_center_3d_right[2]


        msg_filtered.pupil_confidence=lpf.pupil_confidence
        msg_filtered.calib_rotation = lpf.calib_rotation
        msg_filtered.label = lpf.label
        rate.sleep()
        pub1.publish(msg_filtered)
        pub2.publish(msg_filtered2)
        


    rospy.spin()
