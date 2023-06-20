#!/usr/bin/env python
#from catkin_ws.src.tracker.src.pupil_LPF import GAZE_POINT_3D
import rospy
import numpy as np
import math
import time
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from tracker.msg import GazeEuler
from tracker.msg import GazePoint
from tracker.msg import DL_Tracker
import zmq
import math
from tf.transformations import *
import quatornion as q

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox

GAZE_POINT_3D = 1
TEST_EYE = 0

cv_image = np.empty(shape=[0])
cv_image2 = np.empty(shape=[0])

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
        self.calib_rotation = 0.
        self.label = 0

    def callback0(self, msg):
        self.v0 = np.array([msg.point.x, msg.point.y, msg.point.z])

    def callback1(self, msg):
        self.v1 = np.array([msg.point.x, msg.point.y, msg.point.z])

    def callback2(self, msg):
        self.v2 = np.array([msg.point.x, msg.point.y, msg.point.z])

    def callback3(self, msg):
        self.gp = np.array([msg.gaze_point_3d.x, msg.gaze_point_3d.y, msg.gaze_point_3d.z])
        self.calib_rotation = msg.calib_rotation * 3.14 / 180 ## radian  org
        self.calib_rotation_degree = msg.calib_rotation  ## degree
        self.label = msg.label

    def realsense2world(self, point):
        return np.matmul(self.matrix_realsense2world, point) + self.realsense_in_world

#------------edit--------------
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
#--------------------------------------------
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
#------------------------------------------
def color_img_callback(data):
    global cv_image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def depth_img_callback(data):
    global cv_image2
    bridge2 = CvBridge()
    cv_image2 = bridge2.imgmsg_to_cv2(data, "32FC1")

if __name__ == "__main__":

    CALIB = 10
    test_gp = [1.0,0.0,0.0]
    count = 0
    q_orig = quaternion_from_euler(0, 0, 0)

    rospy.init_node('tracker', anonymous=True)
    tracker = Tracker(np.array([0,0,0]))  # real distance for realsense cam
    pub_head = rospy.Publisher('/vector_head', Marker, queue_size=10)
    pub_eye = rospy.Publisher('/vector_eye', Marker, queue_size=10)
    
    pub_eye_raw = rospy.Publisher('/vector_eye_raw', Marker, queue_size=10)

    pub_eye_right = rospy.Publisher('/vector_eye_right', Marker, queue_size=10)
    pub_eye_left = rospy.Publisher('/vector_eye_left', Marker, queue_size=10)

    pub_head_raw = rospy.Publisher('/vector_head_raw', Marker, queue_size=10)
    pub_DL = rospy.Publisher('/dl_tracker', DL_Tracker, queue_size=10)
    rospy.Subscriber("first", PointStamped, tracker.callback0)
    rospy.Subscriber("second", PointStamped, tracker.callback1)
    rospy.Subscriber("third", PointStamped, tracker.callback2)
    rospy.Subscriber("gaze_LPF", GazePoint, tracker.callback3)
    rospy.Subscriber("/bbox2",BoundingBoxArray, callback)

    rospy.Subscriber("/camera/color/image_raw", Image, color_img_callback)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_img_callback)

    with open("/home/kaai/catkin_ws/src/data/head_vector_csv/head_vector.csv","a") as f:            
        f.write(str('unix')+','+str('p_driver[0]')+','+str('p_driver[1]')+','+str('p_driver[2]')+','+str('normal[0]')+ ',' +str('normal[1]')+ ',' +str('normal[2]')+'\n')
    with open("/home/kaai/catkin_ws/src/data/final_vector_csv/final_vector.csv","a") as f:            
        f.write(str('unix')+','+str('p_driver[0]')+','+str('p_driver[1]')+','+str('p_driver[2]')+','+str('v_eye[0]')+ ',' +str('v_eye[1]')+ ',' +str('v_eye[2]')+'\n')
    with open("/home/kaai/catkin_ws/src/data/head_vector_csv/head_vector_angle.csv","a") as f:            
        f.write(str('unix')+','+str('yaw')+','+str('pitch')+ ',' +str('roll')+'\n')
    with open("/home/kaai/catkin_ws/src/data/image/image_info.csv","a") as f:            
        f.write(str('unix')+','+str('count')+'\n')
    with open("/home/kaai/catkin_ws/src/data/depth_image/depth_image_info.csv","a") as f:
        f.write(str('unix')+','+str('count')+'\n')

    while not rospy.is_shutdown():

        line0 = tracker.v1 - tracker.v0
        line1 = tracker.v2 - tracker.v0
        
        r_line0 = tracker.v2 - tracker.v1
        r_line1 = np.array([0, 1, 0])
        normal = np.cross(line0, line1) # current pos of face 3*1 mat
        
        normal[2]=-normal[2] # change z -> -z
        if TEST_EYE:
            normal = np.array([1,0,0])
        if normal[0] < 0:
            normal = -normal


        normal_p = point_pub('normal_p',normal[0]*1000, normal[1]*1000, normal[2]*1000) # vchead_sphere

        tracker.realsense_in_world=np.array([0,0,0])

        v0 = tracker.realsense2world(tracker.v0)
        v1 = tracker.realsense2world(tracker.v1)
        v2 = tracker.realsense2world(tracker.v2)

        # for driver pos 
       
        p_driver = (v0 + v1 + v2) / 3 + np.array([0,0,0.13])

        if TEST_EYE:
            p_driver = np.array([0,0,0])

        marker_normal = make_marker(normal * 999999 + p_driver, p_driver, [0, 0, 1]) #org

#---TODO : get yaw pitch roll , and matmul with v_eye---------------------------------------
        
        q_orig = quaternion_from_euler(0, 0, 0)
        yaw = np.arctan2(normal[1], normal[0])*180/math.pi # ** degree
        pitch =np.arctan2(normal[2], normal[0])*180/math.pi # ** degree
        roll = get_roll(r_line0, r_line1)*180/math.pi # ** degree
        if roll > 90 :
            roll = -(180 - roll)
        q_rot = quaternion_from_euler(-yaw*math.pi/180, pitch*math.pi/180, roll*math.pi/180) # rot angle of face **to radian

        q_orig = quaternion_multiply(q_rot, q_orig) # pose of face

        time.sleep(0.05)
        
        e0 =tuple(v0)
        e1 =tuple(v1)
        v_eye = tracker.gp
        marker_eye_raw = make_marker(v_eye * 999999 +p_driver , p_driver, [0.5, 0, 0])
        v_eye = tuple(tracker.gp) #tracker.gp test_gp
        
        v_eye = q.qv_mult(q_orig,v_eye)
        
        e0 = q.qv_mult(q_orig,e0)
        e1 = q.qv_mult(q_orig,e1)
        
        e0 = np.asarray(e0)
        e1 = np.asarray(e1)
        eye_center =(e0 + e1)/2

        v_eye = np.asarray(v_eye)
        v_eye[0] = -v_eye[0]
        v_eye[1] = -v_eye[1]
        gp_point = point_pub('gp',v_eye[0], v_eye[1], v_eye[2]) # gp_sphere


        marker_eye = make_marker(v_eye * 999999 +p_driver , p_driver, [1, 0, 0])
        marker_eye_right = make_marker(v_eye*10+e0, e0, [0.5, 0.5, 0])
        marker_eye_left = make_marker(v_eye*10+e1, e1, [1, 1, 0])
        #-------------------------------------------
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
#-------------------------------------------------------------------------------------------
        msg_dl = DL_Tracker()
        msg_dl.yaw_eye = np.arctan2(v_eye[1], v_eye[0])
        msg_dl.label = tracker.label

        pub_eye.publish(marker_eye)

        pub_eye_raw.publish(marker_eye_raw)

        pub_eye_right.publish(marker_eye_right)
        pub_eye_left.publish(marker_eye_left)
        
        pub_head_raw.publish(marker_normal)
        pub_DL.publish(msg_dl)

        if p_driver[0]==0 and p_driver[1]==0 and p_driver[2]==0:
            continue
        elif normal[0]==0 and normal[1]==0 and normal[2]==0:
            continue
        elif v_eye[0]==0 and v_eye[1]==0 and normal[2]==0:
            continue
        elif yaw==0 and pitch==0 and roll==0:
            continue

        unix=time.time()
        with open("/home/kaai/catkin_ws/src/data/head_vector_csv/head_vector.csv","a") as f:            
            f.write(str(unix)+','+str(p_driver[0])+','+str(p_driver[1])+','+str(p_driver[2])+','+str(normal[0])+ ',' +str(normal[1])+ ',' +str(normal[2])+'\n')
        with open("/home/kaai/catkin_ws/src/data/final_vector_csv/final_vector.csv","a") as f:            
            f.write(str(unix)+','+str(p_driver[0])+','+str(p_driver[1])+','+str(p_driver[2])+','+str(v_eye[0])+ ',' +str(v_eye[1])+ ',' +str(v_eye[2])+'\n')
        with open("/home/kaai/catkin_ws/src/data/head_vector_csv/head_vector_angle.csv","a") as f:            
            f.write(str(unix)+','+str(yaw)+','+str(pitch)+ ',' +str(roll)+'\n')
        with open("/home/kaai/catkin_ws/src/data/image/image_info.csv","a") as f:            
            f.write(str(unix)+','+str(count)+'\n')
        with open("/home/kaai/catkin_ws/src/data/depth_image/depth_image_info.csv","a") as f:            
            f.write(str(unix)+','+str(count)+'\n')

        if cv_image.size != (640*480*3):
            continue
        if cv_image2.size != (640*480):
            continue
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray,(5,5),0)
        edge_img = cv2.Canny(np.uint8(blur_gray),60,70)

        img_path = '/home/kaai/catkin_ws/src/data/image/'
        img_path2 = '/home/kaai/catkin_ws/src/data/depth_image/'

        if count<10:
            path = img_path + '00000' + str(count) + '.jpg'
            path2 = img_path2 + '00000' + str(count) + '.jpg'
        elif count>9 and count<100:
            path = img_path + '0000' + str(count) + '.jpg'
            path2 = img_path2 + '0000' + str(count) + '.jpg'
        elif count>99 and count<1000:
            path = img_path + '000' + str(count) + '.jpg'
            path2 = img_path2 + '000' + str(count) + '.jpg'
        elif count>999 and count<10000:
            path = img_path + '00' + str(count) + '.jpg'
            path2 = img_path2 + '00' + str(count) + '.jpg'
        elif count>9999 and count<100000:
            path = img_path + '0' + str(count) + '.jpg'
            path2 = img_path2 + '0' + str(count) + '.jpg'
        else:
            path = img_path + str(count) + '.jpg'
            path2 = img_path2 + str(count) + '.jpg'

        count+=1

        cv2.imwrite(path,cv_image)
        cv2.imwrite(path2,cv_image2)


    rospy.spin()
