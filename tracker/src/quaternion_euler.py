#!/usr/bin/env python
from numpy import eye
import rospy
from geometry_msgs.msg import PointStamped, Point
import numpy as np
from tf.transformations import *
import quatornion as q
import math

#rviz x,y,z
point_x=np.array([1,0,0])
point_y=np.array([0,1,0])
point_z=np.array([0,0,1])

class point_pub:
    def __init__(self, name, x, y, z):
        rospy.init_node("xyz")
        self.pub = rospy.Publisher(name, PointStamped, queue_size=10)
        self.data = Point()
        self.data.x, self.data.y, self.data.z = x, y, z
        self.pointstamp = PointStamped()
        self.pointstamp.point = self.data
        self.pointstamp.header.frame_id = "/camera_link"
        self.pub.publish(self.pointstamp)

def get_roll(v1,v2) :
    t1 = np.matmul(v1,v2)
    cos =t1/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.arccos(cos)

eye_exp=np.array([0,0,3])
y_line=np.array([0,1,0])

roll=90*math.pi/180
pitch=0*math.pi/180
yaw=0*math.pi/180

while not rospy.is_shutdown():
    p_x=point_pub("/point_x",point_x[0],point_x[1],point_x[2])
    p_y=point_pub("/point_y",point_y[0],point_y[1],point_y[2])
    p_z=point_pub("/point_z",point_z[0],point_z[1],point_z[2])

    ori_eye=point_pub("original",eye_exp[0],eye_exp[1],eye_exp[2])

    #q_rot=quaternion_from_euler(yaw,pitch,roll)
    q_rot=q.euler_to_quaternion(roll,pitch,yaw)
    #print(q_rot)
    
    a_eye=tuple(eye_exp)
    a_eye=q.qv_mult(q_rot,a_eye)

    a_eye=np.asarray(a_eye)
    a_eye=point_pub("after_eye",a_eye[0],a_eye[1],a_eye[2])
    print(get_roll(eye_exp,y_line)*180/math.pi)

