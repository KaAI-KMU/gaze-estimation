import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np

class HeadposeEstimationModel(nn.Module):
    """Some Information about HeadposeEstimationModel"""
    def __init__(self):
        super(HeadposeEstimationModel, self).__init__()
        self.point1=np.array([0.,0.,0.])
        self.point2=np.array([0.,0.,0.])
        self.point3=np.array([0.,0.,0.])
        self.matrix_realsense2world = np.array([[-1, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 1]])
        self.realsense_in_world = np.array([0.,0.,0.])
        self.point1_mat = np.array([[self.point1[0]],[self.point1[1]],[self.point1[2]]])
        self.point2_mat = np.array([[self.point2[0]],[self.point2[1]],[self.point2[2]]])
        self.point3_mat = np.array([[self.point3[0]],[self.point3[1]],[self.point3[2]]])

    def forward(self, img, depth_img, facial_landmark):
        img = cv2.imread(img)
        img_depth = depth_img
        self.point2[0], self.point2[1] = facial_landmark[36][0:3], facial_landmark[36][4:]
        self.point1[0], self.point1[1] = facial_landmark[42][0:3], facial_landmark[42][4:]
        self.point3[0], self.point3[1] = facial_landmark[48][0:3], facial_landmark[48][4:]
        self.point2[2] = img_depth[int(self.point2[1])][int(self.point2[0])]
        self.point1[2] = img_depth[int(self.point1[1])][int(self.point1[0])]
        self.point3[2] = img_depth[int(self.point3[1])][int(self.point3[0])]

        self.projection()
        self.point1 = self.realsense2world(self.point1)
        self.point2 = self.realsense2world(self.point2)
        self.point3 = self.realsense2world(self.point3)
        normal_line1 = self.point3 - self.point2
        normal_line2 = self.point1 - self.point2
        normal = np.cross(normal_line1, normal_line2)
        head_pose = normal
        return head_pose
    
    def projection(self):
        self.point1_mat = np.array([[self.point1[0]*self.point1[2]],[self.point1[1]*self.point1[2]],[self.point1[2]]])
        self.point2_mat = np.array([[self.point2[0]*self.point2[2]],[self.point2[1]*self.point2[2]],[self.point2[2]]])
        self.point3_mat = np.array([[self.point3[0]*self.point3[2]],[self.point3[1]*self.point3[2]],[self.point3[2]]])
        p_mat = np.array([[0,0,1],[-0.00160981,0,0.5177980],[0,-0.00160981,0.390396]])
        mat1 = np.dot(p_mat,self.point1_mat)
        mat2 = np.dot(p_mat,self.point2_mat)
        mat3 = np.dot(p_mat,self.point3_mat)
        self.point1[0], self.point1[1], self.point1[2] = mat1[0][0]*0.001, mat1[1][0]*0.001, mat1[2][0]*0.001
        self.point2[0], self.point2[1], self.point2[2] = mat2[0][0]*0.001, mat2[1][0]*0.001, mat2[2][0]*0.001
        self.point3[0], self.point3[1], self.point3[2] = mat3[0][0]*0.001, mat3[1][0]*0.001, mat3[2][0]*0.001

    def realsense2world(self, point): #좌표계의 변환
        return np.matmul(self.matrix_realsense2world, point) + self.realsense_in_world