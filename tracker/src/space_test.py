#!/usr/bin/env python
#from catkin_ws.src.tracker.src.pupil_LPF import GAZE_POINT_3D
import rospy
import numpy as np
import math
import time
from std_msgs.msg import Float64
from std_msgs.msg import Int64

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pynput import keyboard
import pandas as pd

space = 5

#Keyboard Listener
def on_press(key):
    return True

def on_release(key):
    global space
    if key == keyboard.Key.space:
        space = 100
        return False
    elif key == keyboard.Key.esc:
        space = 500
        return False
    else:
        space = 5
        return False

rospy.init_node("space_bar_check_node", anonymous=True)
space_bar = rospy.Publisher("/space_dataset", Int64, queue_size=10)

if __name__ == "__main__":
    while not rospy.is_shutdown():
        with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
            listener.join()
            space_bar.publish(space)
