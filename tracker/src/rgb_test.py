#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import time 

cv_image = np.empty(shape=[0])

def callback(data):
    global cv_image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

if __name__ == "__main__":
    rospy.init_node('test', anonymous=True)
    rospy.Subscriber('/zed/zed_node/rgb/image_rect_color',Image, callback)
    count = 1

    time.sleep(1)
    while not rospy.is_shutdown():
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray,(5,5),0)
        edge_img = cv2.Canny(np.uint8(blur_gray),60,70)

        path = '/home/kaai/Datasets/KaAI/zed/'
        img = path + str(count)+'.jpg'
        print(count)
        count += 1
        cv2.imwrite(img,cv_image)
    rospy.spin()