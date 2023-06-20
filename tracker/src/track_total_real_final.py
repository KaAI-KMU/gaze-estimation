#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#from catkin_ws.src.tracker.src.pupil_LPF import GAZE_POINT_3D
from cmath import sqrt
import rospy
import numpy as np
import math

import time
from std_msgs.msg import Int64
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from tracker.msg import GazeEuler
from tracker.msg import GazePoint
from tracker.msg import gazepoint #cw
from tracker.msg import DL_Tracker

import zmq
from tf.transformations import *
import quatornion as q
import pandas as pd

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox

BoundingBoxCheck = False

space_check = False
esc_check = False
count = 0
cv_image = np.empty(shape=[0])
cv_image2 = np.empty(shape=[0])
cv_image3 = np.empty(shape=[0])
cv_image4 = np.empty(shape=[0])
cv_image5 = np.empty(shape=[0])

def make_marker(pt1, pt2, color):
    marker = Marker()
    marker.header.frame_id = "/camera_link"
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.ns = 'a'
    marker.id = 0
    marker.scale.x = 0.05
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    p1, p2 = Point(), Point()
    p1.x = pt1[0]
    p1.y = pt1[1]
    p1.z = pt1[2]
    p2.x = pt2[0]
    p2.y = pt2[1]
    p2.z = pt2[2]
    marker.points.append(p1)
    marker.points.append(p2)
    return marker

class Tracker:
    def __init__(self, realsense_in_world):
        self.matrix_realsense2world = np.array([[-1, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 1]])
        self.realsense_in_world = realsense_in_world

        self.v0 = np.array([0, 0, 0])
        self.v1 = np.array([0, 0, 0])
        self.v2 = np.array([0, 0, 0])
        self.r0 = np.array([0, 0, 0])
        self.r1 = np.array([0, 0, 0])        
        self.gp = np.array([0, 0, 0])
        self.gaze_normal_left = np.array([0,0,0])
        self.gaze_normal_right = np.array([0,0,0])
        self.eye_center_left = np.array([0,0,0])
        self.eye_center_right = np.array([0,0,0])
        self.pupil_confidence = 0.
        #self.calib_rotation = 0.
        # self.label = 0
        self.norm_pos=np.array([0,0])
        self.surface_name = ""
        self.gaze_on_surface = ""
        self.surface_norm_pos = np.array([0,0])
        self.surfaces_data_d = np.array([0,0,0])
        self.base_data_id0 = np.array([0,0])
        self.base_data_id1 = np.array([0,0])

    def callback0(self, msg):
        self.v0 = np.array([msg.point.x, msg.point.y, msg.point.z])

    def callback1(self, msg):
        self.v1 = np.array([msg.point.x, msg.point.y, msg.point.z])

    def callback2(self, msg):
        self.v2 = np.array([msg.point.x, msg.point.y, msg.point.z])

    def callback3(self, msg):
        self.gp = np.array([msg.gaze_point_3d.x, msg.gaze_point_3d.y, msg.gaze_point_3d.z])
        self.gaze_normal_left = np.array([msg.gaze_normals_3d_left.x,msg.gaze_normals_3d_left.y,msg.gaze_normals_3d_left.z])
        self.gaze_normal_right = np.array([msg.gaze_normals_3d_right.x,msg.gaze_normals_3d_right.y,msg.gaze_normals_3d_right.z])
        self.eye_center_left = np.array([msg.eye_centers_3d_left.x,msg.eye_centers_3d_left.y,msg.eye_centers_3d_left.z])
        self.eye_center_right = np.array([msg.eye_centers_3d_right.x,msg.eye_centers_3d_right.y,msg.eye_centers_3d_right.z])
        #self.calib_rotation = msg.calib_rotation * 3.14 / 180
        self.norm_pos=np.array([msg.norm_pos.x,msg.norm_pos.y])
        #self.calib_rotation_degree = msg.calib_rotation 
        # self.label = msg.label
        self.surface_name = msg.surface_name
        self.gaze_on_surface = msg.gaze_on_surf
        self.surface_norm_pos = np.array([msg.s_norm_gp.x,msg.s_norm_gp.y])
        self.surfaces_data_d = msg.surfaces_data_d
        self.base_data_id0 = msg.base_data_id0
        self.base_data_id1 = msg.base_data_id1
        if msg.pupil_confidence > 0.8:
            self.pupil_confidence=msg.pupil_confidence
        else:
            self.pupil_confidence=0

    def realsense2world(self, point):
        return np.matmul(self.matrix_realsense2world, point) + self.realsense_in_world

class point_pub:
    def __init__(self, name, x, y, z):
        self.pub = rospy.Publisher(name, PointStamped, queue_size=10)
        self.data = Point()
        self.data.x, self.data.y, self.data.z = x, y, z
        self.pointstamp = PointStamped()
        self.pointstamp.point = self.data
        self.pointstamp.header.frame_id = "/camera_link" # change velodyne to camera_link
        self.pub.publish(self.pointstamp)

def get_roll(v1,v2) :
    t1 = np.dot(v1,v2)
    cos =t1/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.arccos(cos)

#Image Reader
def color_img_callback(data):
    global cv_image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def depth_img_callback(data):
    global cv_image2
    bridge2 = CvBridge()
    cv_image2 = bridge2.imgmsg_to_cv2(data, "32FC1")

def fov_img_callback(data):
    global cv_image3
    bridge3 = CvBridge()
    cv_image3 = bridge3.imgmsg_to_cv2(data, "bgr8")

def front_img_callback(data):
    global cv_image4
    bridge4 = CvBridge()
    cv_image4 = bridge4.imgmsg_to_cv2(data, "bgr8")

def front_depth_img_callback(data):
    global cv_image5
    bridge5 = CvBridge()
    cv_image5 = bridge5.imgmsg_to_cv2(data, "32FC1")

box_1=BoundingBox()
box_2=BoundingBox()
box_3=BoundingBox()
box_4=BoundingBox()
box_5=BoundingBox()
box_6=BoundingBox()

class bbox_boundary:
    def __init__(self,bbox):
        self.bbox=bbox
        self.p1=np.array([0,0,0])
        self.p2=np.array([1,1,1])
        self.p3=np.array([0,0,0])
        self.p4=np.array([0,0,0])
        self.p5=np.array([0,0,0])
        self.p6=np.array([0,0,0])
        self.p7=np.array([0,0,0])
        self.p8=np.array([0,0,0])
        self.label=0
        self.check_point_x=0
        self.cehck_point_y=0
        self.cross_z=0
        self.slope=1
        self.z_slope=1

    def bbox_edge(self,c_x,c_y,c_z,d_x,d_y,d_z):
        self.p1=np.array([c_x-d_x/2,c_y+d_y/2,c_z-d_z/2])
        self.p2=np.array([c_x-d_x/2,c_y-d_y/2,c_z-d_z/2])
        self.p5=np.array([c_x-d_x/2,c_y+d_y/2,c_z+d_z/2])

    def label_check(self,label):
        if label==1:
            self.label="handle"
        elif label==2:
            self.label="instrument_panel"
        elif label==3:
            self.label="left_side_mirror"
        elif label==4:
            self.label="right_side_mirror"
        elif label==5:
            self.label="gear_stick"
        elif label==6:
            self.label="center_fascia"

    def check_in_bbox(self,start_,finish_):
        max_slope=(self.p1[1]-start_marker[1])/(self.p1[0]-start_marker[0])
        min_slope=(self.p2[1]-start_marker[1])/(self.p2[0]-start_marker[0])
        slope=(finish_marker[1]-start_marker[1])/(finish_marker[0]-start_marker[0])
        if min_slope<=slope and slope<=max_slope:
            z_min_slope=(self.p1[2]-start_marker[2])/(self.p1[0]-start_marker[0])
            z_max_slope=(self.p5[2]-start_marker[2])/(self.p5[0]-start_marker[0])
            z_slope=(finish_marker[2]-start_marker[2])/(finish_marker[0]-start_marker[0])
            if  z_min_slope <= z_slope and z_slope <= z_max_slope:
                print("You currently watching ",self.label)


def callback(boundingboxarray):
    global box_1, box_2, box_3, box_4, box_5, box_6
    box_1=boundingboxarray.boxes[0]
    box_2=boundingboxarray.boxes[1]
    box_3=boundingboxarray.boxes[2]
    box_4=boundingboxarray.boxes[3]
    box_5=boundingboxarray.boxes[4]
    box_6=boundingboxarray.boxes[5]

def callback_space(space):
    global space_check, esc_check
    if space.data == 100:
        space_check = True
    elif space.data ==500:
        esc_check = True
    else:
        space_check = False

title_data1 = {'a':['UNIX_time'],'b':['confidence'],'c1':['l_eye_center_x'],'c2':['l_eye_center_y'],'c3':['l_eye_center_z'],'d1':['r_eye_center_x'],'d2':['r_eye_center_y'],'d3':['r_eye_center_z'],'e1':['l_eye_dir_x'],'e2':['l_eye_dir_y'],'e3':['l_eye_dir_z'],'f1':['r_eye_dir_x'],'f2':['r_eye_dir_y'],'f3':['r_eye_dir_z'],'g1':['gaze_point_x'],'g2':['gaze_point_y'],'g3':['gaze_point_z'],'h1':['norm_pos_x'],'h2':['norm_pos_y'],'i':['surface_name'],'j1':['surface_norm_pos_x'],'j2':['surface_norm_pos_y'],'k':['gaze_on_surface']}
title_data2 = {'a':['UNIX_time'],'b':['confidence'],'c1':['gaze_dir_avg_x'],'c2':['gaze_dir_avg_y'],'c3':['gaze_dir_avg_z'],'d':['l_gaze_phi'],'e':['l_gaze_theta'],'f':['r_gaze_phi'],'g':['r_gaze_theta'],'h':['gaze_phi_avg'],'i':['gaze_theta_avg']}
title_data3 = {'a':['UNIX_time'],'b':['confidence'],'c1':['head_vec_x'],'c2':['head_vec_y'],'c3':['head_vec_z'],'d':['head_roll'],'e':['head_pitch'],'f':['head_yaw'],'g1':['l_eye_dir_x'],'g2':['l_eye_dir_y'],'g3':['l_eye_dir_z'],'h1':['r_eye_dir_x'],'h2':['r_eye_dir_y'],'h3':['r_eye_dir_z'],'i1':['gaze_dir_avg_x'],'i2':['gaze_dir_avg_y'],'i3':['gaze_dir_avg_z'],'j1':['p_driver_x'],'j2':['p_driver_y'],'j3':['p_driver_z'],'k1':['camera_orig_x'],'k2':['camera_orig_y'],'k3':['camera_orig_z']}
title_data4 = {'a':['surface_data_d'],'b':['base-data_id0'],'c':['base-data_id1']}


df1=pd.DataFrame(title_data1)
df1.to_csv('/home/kaai/Datasets/KaAI/csv/raw_data.csv', mode='a', index=False, header=False)
df2=pd.DataFrame(title_data2)
df2.to_csv('/home/kaai/Datasets/KaAI/csv/eyetracker.csv', mode='a', index=False, header=False)
df3=pd.DataFrame(title_data3)
df3.to_csv('/home/kaai/Datasets/KaAI/csv/driver.csv', mode='a', index=False, header=False)
df4=pd.DataFrame(title_data4)
df4.to_csv('/home/kaai/Datasets/KaAI/csv/trash.csv', mode='a', index=False, header=False)

if __name__ == "__main__":
    CALIB = 10
    test_gp = [0.0,1.0,0.0]
    count = 1
    q_orig = quaternion_from_euler(0, 0, 0)  

    rospy.init_node('tracker', anonymous=True)

    tracker = Tracker(np.array([0,0,0]))  # real distance for realsense cam
    pub_head = rospy.Publisher('/vector_head', Marker, queue_size=10)
    pub_eye = rospy.Publisher('/vector_eye', Marker, queue_size=10)    

    pub_eye_raw = rospy.Publisher('/vector_eye_raw', Marker, queue_size=10)  #gaze_point_3d
    pub_head_raw = rospy.Publisher('/vector_head_raw', Marker, queue_size=10)

    pub_direction_left_raw = rospy.Publisher('/vector_direction_left_raw', Marker, queue_size=10)  #gaze_point_3d
    pub_direction_right_raw = rospy.Publisher('/vector_direction_right_raw', Marker, queue_size=10)
 
    pub_direction_left = rospy.Publisher('/vector_direction_left', Marker, queue_size=10)  #gaze_point_3d
    pub_direction_right = rospy.Publisher('/vector_direction_right', Marker, queue_size=10)


    rospy.Subscriber("first", PointStamped, tracker.callback0)
    rospy.Subscriber("second", PointStamped, tracker.callback1)
    rospy.Subscriber("third", PointStamped, tracker.callback2)
    rospy.Subscriber("gaze_LPF", GazePoint, tracker.callback3)

    rospy.Subscriber("/camera/color/image_raw", Image, color_img_callback)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_img_callback)
    rospy.Subscriber("/FOV", Image, fov_img_callback)

    rospy.Subscriber("/space_dataset", Int64, callback_space)

    rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, front_img_callback)
    rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, front_depth_img_callback)

    print("Start Calibration...")
    time.sleep(1)

    print("__Success__")

    while not rospy.is_shutdown():
        line0 = tracker.v1 - tracker.v0
        line1 = tracker.v2 - tracker.v0        
        r_line0 = tracker.v0 - tracker.v1 #cw
        r_line1 = np.array([0, 1, 0])
        tracker.realsense_in_world=np.array([0,0,0])

        v0 = tracker.realsense2world(tracker.v0)
        v1 = tracker.realsense2world(tracker.v1)
        v2 = tracker.realsense2world(tracker.v2)
        normal_line1=v2-v1
        normal_line2=v0-v1
        normal = np.cross(normal_line1, normal_line2)
        normal = -normal

        vv00 = point_pub('vv00',v0[0], v0[1], v0[2])
        vv11 = point_pub('vv11',v1[0], v1[1], v1[2])
        vv22 = point_pub('vv22',v2[0], v2[1], v2[2])

        camera_org = np.array([v1[0]+(v0[0]-v1[0])*6.5/11.3, v1[1]+(v0[1]-v1[1])*6.5/11.3, (v0[2]+v1[2])/2-(v2[2]-(v0[2]+v1[2])/2)*3/7])
        camera_origin = point_pub('camera_origin', camera_org[0], camera_org[1], camera_org[2])

        # for driver pos        
        p_driver = (v0 + v1 + v2) / 3

        marker_normal = make_marker(normal * 1000 + p_driver, p_driver, [0, 0, 1]) #org


###################




###################


#----------yaw,pitch,roll----------#
        q_orig = quaternion_from_euler(0, 0, 0)
        yaw = np.arctan2(normal[1], normal[0])*180/math.pi # ** degree
        pitch =np.arctan2(normal[2], math.sqrt(normal[0]*normal[0]+normal[1]*normal[1]))*180/math.pi # ** degree    

        r_line00=np.array(v1-v0)        
        final_line=[0,r_line00[1],r_line00[2]]
        final_line1=[0,r_line00[1],0]     

        if final_line[2]>0:
            roll=get_roll(final_line,final_line1)*180/math.pi 
        else:
            roll=-(get_roll(final_line,final_line1)*180/math.pi)
        #print('yaw,pitch,roll :', yaw, -pitch, -roll)

#----------quarternion rotation----------#

        q_rot = q.euler_to_quaternion(0, -pitch*(math.pi/180), yaw*(math.pi/180)) 
        #q_rot = q.euler_to_quaternion(-roll*(math.pi/180), -pitch*(math.pi/180), yaw*(math.pi/180)) 
        q_orig = quaternion_multiply(q_rot, q_orig) # pose of face
        time.sleep(0.05)
        
        v_eye = tracker.gp

        marker_eye_raw = make_marker(v_eye * 1000 +p_driver , p_driver, [0.5, 0, 0])
        v_eye = tuple(tracker.gp) #tracker.gp
        v_eye = q.qv_mult(q_rot,v_eye)

        v_eye = np.asarray(v_eye)
        norm_pos=tuple(tracker.norm_pos)
        gp_point = point_pub('gp',v_eye[0], v_eye[1], v_eye[2]) # gp_sphere
        marker_eye = make_marker(v_eye * 1000 +p_driver , p_driver, [1, 0, 0])

        ##########################
        v_left_eye_direction = tuple(tracker.gaze_normal_left)
        v_right_eye_direction = tuple(tracker.gaze_normal_right)
        left_eye_center = tracker.eye_center_left/1000
        right_eye_center = tracker.eye_center_right/1000

        v_left_eye_direction_raw = tracker.gaze_normal_left
        v_right_eye_direction_raw = tracker.gaze_normal_right

        gaze_direction_average = (tracker.gaze_normal_left+tracker.gaze_normal_right)/2.
        gaze_phi_left=math.asin(tracker.gaze_normal_left[1])
        gaze_theta_left=math.atan2(tracker.gaze_normal_left[0],tracker.gaze_normal_left[2])
        gaze_phi_right=math.asin(tracker.gaze_normal_right[1])
        gaze_theta_right=math.atan2(tracker.gaze_normal_right[0],tracker.gaze_normal_right[2])

        gaze_phi_average = (gaze_phi_left+gaze_phi_right)/2.
        gaze_theta_average = (gaze_theta_left+gaze_theta_right)/2.

        v_left_eye_direction = q.qv_mult(q_rot,v_left_eye_direction)
        v_left_eye_direction = np.asarray(v_left_eye_direction)

        v_right_eye_direction = q.qv_mult(q_rot,v_right_eye_direction)
        v_right_eye_direction = np.asarray(v_right_eye_direction)

        confidence=tracker.pupil_confidence

        marker_direction_left_raw = make_marker(v_left_eye_direction_raw * 1000 +camera_org+left_eye_center , camera_org+left_eye_center, [1, 0, 0])
        marker_direction_right_raw = make_marker(v_right_eye_direction_raw * 1000 +camera_org+right_eye_center , camera_org+right_eye_center, [1, 0, 0])

        pub_direction_left.publish(marker_direction_left_raw)
        pub_direction_right.publish(marker_direction_right_raw)

        marker_direction_left = make_marker(v_left_eye_direction * 1000 +camera_org+left_eye_center , camera_org+left_eye_center, [1, 0, 0])
        marker_direction_right = make_marker(v_right_eye_direction * 1000 +camera_org+right_eye_center , camera_org+right_eye_center, [1, 0, 0])

        pub_direction_left.publish(marker_direction_left)
        pub_direction_right.publish(marker_direction_right)

        gaze_direction_average_world_coordinate = (v_left_eye_direction+v_right_eye_direction)/2.
        ##############################
        #-------------------------------------------
        if BoundingBoxCheck:
            start_marker=np.array([marker_eye.points[1].x,marker_eye.points[1].y,marker_eye.points[1].z])
            finish_marker=np.array([marker_eye.points[0].x,marker_eye.points[0].y,marker_eye.points[0].z])

            box_1_boundary=bbox_boundary(box_1)
            box_2_boundary=bbox_boundary(box_2)
            box_3_boundary=bbox_boundary(box_3)
            box_4_boundary=bbox_boundary(box_4)
            box_5_boundary=bbox_boundary(box_5)
            box_6_boundary=bbox_boundary(box_6)

            box_1_boundary.bbox_edge(box_1.pose.position.x,box_1.pose.position.y,box_1.pose.position.z,box_1.dimensions.x,box_1.dimensions.y,box_1.dimensions.z)
            box_2_boundary.bbox_edge(box_2.pose.position.x,box_2.pose.position.y,box_2.pose.position.z,box_2.dimensions.x,box_2.dimensions.y,box_2.dimensions.z)
            box_3_boundary.bbox_edge(box_3.pose.position.x,box_3.pose.position.y,box_3.pose.position.z,box_3.dimensions.x,box_3.dimensions.y,box_3.dimensions.z)
            box_4_boundary.bbox_edge(box_4.pose.position.x,box_4.pose.position.y,box_4.pose.position.z,box_4.dimensions.x,box_4.dimensions.y,box_4.dimensions.z)
            box_5_boundary.bbox_edge(box_5.pose.position.x,box_5.pose.position.y,box_5.pose.position.z,box_5.dimensions.x,box_5.dimensions.y,box_5.dimensions.z)
            box_6_boundary.bbox_edge(box_6.pose.position.x,box_6.pose.position.y,box_6.pose.position.z,box_6.dimensions.x,box_6.dimensions.y,box_6.dimensions.z)

            box_1_boundary.label_check(box_1.label)
            box_2_boundary.label_check(box_2.label)
            box_3_boundary.label_check(box_3.label)
            box_4_boundary.label_check(box_4.label)
            box_5_boundary.label_check(box_5.label)
            box_6_boundary.label_check(box_6.label)

            box_1_boundary.check_in_bbox(start_marker,finish_marker)
            box_2_boundary.check_in_bbox(start_marker,finish_marker)
            box_3_boundary.check_in_bbox(start_marker,finish_marker)
            box_4_boundary.check_in_bbox(start_marker,finish_marker)
            box_5_boundary.check_in_bbox(start_marker,finish_marker)
            box_6_boundary.check_in_bbox(start_marker,finish_marker)
        #------------------------------------------------
        msg_dl = DL_Tracker()
        msg_dl.yaw_eye = np.arctan2(v_eye[1], v_eye[0])
        #msg_dl.label = tracker.label

        pub_eye.publish(marker_eye)
        pub_eye_raw.publish(marker_eye_raw)
        pub_head_raw.publish(marker_normal)
        #-------------------------------------------------------------------------------------------
        # After put spacebar key, acquire dataset
        # if p_driver[0]==0 and p_driver[1]==0 and p_driver[2]==0:
        #     continue
        # elif normal[0]==0 and normal[1]==0 and normal[2]==0:
        #     continue
        # elif v_eye[0]==0 and v_eye[1]==0 and normal[2]==0:
        #     continue
        # elif yaw==0 and pitch==0 and roll==0:
        #     continue
        unix = time.time()

        eye_yaw=np.arctan2(v_eye[1],v_eye[0])*180/math.pi
        eye_pitch =np.arctan2(v_eye[2], math.sqrt(v_eye[0]*v_eye[0]+v_eye[1]*v_eye[1]))*180/math.pi # ** degree    
        eye_roll=roll
        
        if space_check:
            normal_value = math.sqrt(math.pow(normal[0],2)+math.pow(normal[1],2)+math.pow(normal[2],2))
            eye_value = math.sqrt(math.pow(v_eye[0],2)+math.pow(v_eye[1],2)+math.pow(v_eye[2],2))
            
            data1 = {'a':[unix],'b':[confidence],'c1':[left_eye_center[0]],'c2':[left_eye_center[1]],'c3':[left_eye_center[2]],'d1':[right_eye_center[0]],'d2':[right_eye_center[1]],'d3':[right_eye_center[2]],'e1':[tracker.gaze_normal_left[0]],'e2':[tracker.gaze_normal_left[1]],'e3':[tracker.gaze_normal_left[2]],'f1':[tracker.gaze_normal_right[0]],'f2':[tracker.gaze_normal_right[1]],'f3':[tracker.gaze_normal_right[2]],'g1':[tracker.gp[0]],'g2':[tracker.gp[1]],'g3':[tracker.gp[2]],'h1':[tracker.norm_pos[0]],'h2':[tracker.norm_pos[1]],'i':[tracker.surface_name],'j1':[tracker.surface_norm_pos[0]],'j2':[tracker.surface_norm_pos[1]],'k':[tracker.gaze_on_surface]}
            data2 = {'a':[unix],'b':[confidence],'c1':[gaze_direction_average[0]],'c2':[gaze_direction_average[1]],'c3':[gaze_direction_average[2]],'d':[gaze_phi_left],'e':[gaze_theta_left],'f':[gaze_phi_right],'g':[gaze_theta_right],'h':[gaze_phi_average],'i':[gaze_theta_average]}
            data3 = {'a':[unix],'b':[confidence],'c1':[normal[0]],'c2':[normal[1]],'c3':[normal[2]],'d':[roll],'e':[pitch],'f':[yaw],'g1':[v_left_eye_direction[0]],'g2':[v_left_eye_direction[2]],'g3':[v_left_eye_direction[2]],'h1':[v_right_eye_direction[0]],'h2':[v_right_eye_direction[1]],'h3':[v_right_eye_direction[2]],'i1':[gaze_direction_average_world_coordinate[0]],'i2':[gaze_direction_average_world_coordinate[1]],'i3':[gaze_direction_average_world_coordinate[2]],'j1':[p_driver[0]],'j2':[p_driver[1]],'j3':[p_driver[2]],'k1':[camera_org[0]],'k2':[camera_org[1]],'k3':[camera_org[2]]}
            data4 = {'a':[tracker.surfaces_data_d],'b':[tracker.base_data_id0],'c':[tracker.base_data_id1]}

            df1=pd.DataFrame(data1)
            df1.to_csv('/home/kaai/Datasets/KaAI/csv/raw_data.csv', mode='a', index=False, header=False)
            df2=pd.DataFrame(data2)
            df2.to_csv('/home/kaai/Datasets/KaAI/csv/eyetracker.csv', mode='a', index=False, header=False)
            df3=pd.DataFrame(data3)
            df3.to_csv('/home/kaai/Datasets/KaAI/csv/driver.csv', mode='a', index=False, header=False)
            df4=pd.DataFrame(data4)
            df4.to_csv('/home/kaai/Datasets/KaAI/csv/trash.csv', mode='a', index=False, header=False)

            with open('/home/kaai/Datasets/KaAI/bin/depth_{}.bin'.format(count),'a') as f:
                for i in range(cv_image2.shape[0]):
                    for j in range(cv_image2.shape[1]):
                        f.write(str(cv_image2[i][j])+' ')
                    f.write('\n')

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            blur_gray = cv2.GaussianBlur(gray,(5,5),0)
            edge_img = cv2.Canny(np.uint8(blur_gray),60,70)

            img_path = '/home/kaai/Datasets/KaAI/rgb_img/'
            img_path2 ='/home/kaai/Datasets/KaAI/depth_img/'
            img_path3 ='/home/kaai/Datasets/KaAI/fov_img/'
            img_path4 ='/home/kaai/Datasets/KaAI/front_img/'
            img_path5 ='/home/kaai/Datasets/KaAI/front_depth_img/'

            if count<10:
                path = img_path + '00000' + str(count) + '.jpg'
                path2 = img_path2 + '00000' + str(count) + '.jpg'
                path3 = img_path3 + '00000' + str(count) + '.jpg'
                path4 = img_path4 + '00000' + str(count) + '.jpg'
                path5 = img_path5 + '00000' + str(count) + '.jpg'
            elif count>9 and count<100:
                path = img_path + '0000' + str(count) + '.jpg'
                path2 = img_path2 + '0000' + str(count) + '.jpg'
                path3 = img_path3 + '0000' + str(count) + '.jpg'
                path4 = img_path4 + '0000' + str(count) + '.jpg'
                path5 = img_path5 + '0000' + str(count) + '.jpg'
            elif count>99 and count<1000:
                path = img_path + '000' + str(count) + '.jpg'
                path2 = img_path2 + '000' + str(count) + '.jpg'
                path3 = img_path3 + '000' + str(count) + '.jpg'
                path4 = img_path4 + '000' + str(count) + '.jpg'
                path5 = img_path5 + '000' + str(count) + '.jpg'
            elif count>999 and count<10000:
                path = img_path + '00' + str(count) + '.jpg'
                path2 = img_path2 + '00' + str(count) + '.jpg'
                path3 = img_path3 + '00' + str(count) + '.jpg'
                path4 = img_path4 + '00' + str(count) + '.jpg'
                path5 = img_path5 + '00' + str(count) + '.jpg'
            elif count>9999 and count<100000:
                path = img_path + '0' + str(count) + '.jpg'
                path2 = img_path2 + '0' + str(count) + '.jpg'
                path3 = img_path3 + '0' + str(count) + '.jpg'
                path4 = img_path4 + '0' + str(count) + '.jpg'
                path5 = img_path5 + '0' + str(count) + '.jpg'
            else:
                path = img_path + str(count) + '.jpg'
                path2 = img_path2 + str(count) + '.jpg'
                path3 = img_path3 + str(count) + '.jpg'
                path4 = img_path4 + str(count) + '.jpg'
                path5 = img_path5 + str(count) + '.jpg'

            count+=1
            
            cv2.imwrite(path,cv_image)
            cv2.imwrite(path2,cv_image2)
            cv2.imwrite(path3,cv_image3)
            cv2.imwrite(path4,cv_image4)
            cv2.imwrite(path5,cv_image5)
            print(count)
            if esc_check:
                print('esc')
                space_check = False
                esc_check = False
        #-------------------------------------------------------------------------------------------


    rospy.spin()
