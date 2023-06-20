#!/usr/bin/env python

import rospy
#from ???.msg import Sync


class DataSync:

    def __init__(self):
        self.pub_msg = Sync()

    def callback_niro(self, msg):
        self.pub_msg.CAN_vehicle_speed =
        self.pub_msg.CAN_throttle_position =
        self.pub_msg.CAN_steering_angle =
        self.pub_msg.CAN_engine_rpm =
        self.pub_msg.CAN_steering_velocity =
        self.pub_msg.CAN_turn_signal =
        self.pub_msg.CAN_LDWS_warning =

    def callback_mobileye(self, msg):
        self.pub_msg.CAN_distance_to_left_lane =
        self.pub_msg.CAN_distance_to_right_lane =
        self.pub_msg.CAN_crossing_event_left_or_right =
        self.pub_msg.CAN_lane_curv =

    def callback_tracker(self, msg):
        self.pub_msg.Tracker_head_yaw =
        self.pub_msg.Tracker_eye_yaw =
        self.pub_msg.LABEL =

    def callback_lidar(self, msg):
        self.pub_msg.LF_x =
        self.pub_msg.LF_y =
        self.pub_msg.LM_x =
        self.pub_msg.LM_y =
        self.pub_msg.LB_x =
        self.pub_msg.LB_y =
        self.pub_msg.CF_x =
        self.pub_msg.CF_y =
        self.pub_msg.CB_x =
        self.pub_msg.CB_y =
        self.pub_msg.RF_x =
        self.pub_msg.RF_y =
        self.pub_msg.RM_x =
        self.pub_msg.RM_y =
        self.pub_msg.RB_x =
        self.pub_msg.RB_y =


if __name__ == '__main__':
    sync = DataSync()
    rospy.init_node('data_sync', anonymous=True)
    pub = rospy.Publisher('sync', Sync, queue_size=10)

    rospy.Subscriber('niro_can', Niro, lpf.callback_niro)
    rospy.Subscriber('mobileye_can', Mobileye, lpf.callback_mobileye)
    rospy.Subscriber('dl_tracker', DL_Tracker, lpf.callback_tracker)
    #rospy.Subscriber('???', ???, lpf.callback_lidar)

    while not rospy.is_shutdown():
        msg_filtered = GazePoint()
        if GAZE_POINT_3D:
            msg_filtered.gaze_point_3d.x = lpf.filterValue_3d[0]
            msg_filtered.gaze_point_3d.y = lpf.filterValue_3d[1]
            msg_filtered.gaze_point_3d.z = lpf.filterValue_3d[2]
        if NORM_POS:
            msg_filtered.norm_pos.x = lpf.filterValue_norm[0]
            msg_filtered.norm_pos.y = lpf.filterValue_norm[1]
        msg_filtered.calib_rotation = lpf.calib_rotation
        msg_filtered.label = lpf.label
        pub.publish(msg_filtered)
    rospy.spin()















