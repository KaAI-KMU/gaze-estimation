import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
import os
import pandas as pd

class HeadposeEstimationModel():
    def __init__(self):
        self.point1=np.array([0.,0.,0.])
        self.point2=np.array([0.,0.,0.])
        self.point3=np.array([0.,0.,0.])
        self.point4=np.array([0.,0.,0.])
        self.point5=np.array([0.,0.,0.])
        self.point6=np.array([0.,0.,0.])
        self.matrix_realsense2world = np.array([[-1, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, 1]])
        self.realsense_in_world = np.array([0.,0.,0.])
        self.point1_mat = np.array([[self.point1[0]],[self.point1[1]],[self.point1[2]]])
        self.point2_mat = np.array([[self.point2[0]],[self.point2[1]],[self.point2[2]]])
        self.point3_mat = np.array([[self.point3[0]],[self.point3[1]],[self.point3[2]]])
        self.point4_mat = np.array([[self.point4[0]],[self.point4[1]],[self.point4[2]]])
        self.point5_mat = np.array([[self.point5[0]],[self.point5[1]],[self.point5[2]]])
        self.point6_mat = np.array([[self.point6[0]],[self.point6[1]],[self.point6[2]]])

    def forward(self, depth_img, facial_landmark):
        img_depth = depth_img
        self.point1[0], self.point1[1] = facial_landmark[54][0:3], facial_landmark[54][4:]
        self.point2[0], self.point2[1] = facial_landmark[48][0:3], facial_landmark[48][4:]
        self.point3[0], self.point3[1] = facial_landmark[33][0:3], facial_landmark[33][4:]
        self.point4[0], self.point4[1] = facial_landmark[8][0:3], facial_landmark[8][4:]
        self.point5[0], self.point5[1] = facial_landmark[36][0:3], facial_landmark[36][4:]
        self.point6[0], self.point6[1] = facial_landmark[45][0:3], facial_landmark[45][4:]
        self.point1[2] = img_depth[int(self.point1[1])][int(self.point1[0])]
        self.point2[2] = img_depth[int(self.point2[1])][int(self.point2[0])]
        self.point3[2] = img_depth[int(self.point3[1])][int(self.point3[0])]
        self.point4[2] = img_depth[int(self.point4[1])][int(self.point4[0])]
        self.point5[2] = img_depth[int(self.point5[1])][int(self.point5[0])]
        self.point6[2] = img_depth[int(self.point6[1])][int(self.point6[0])]

        self.projection()
        self.point1 = self.realsense2world(self.point1)
        self.point2 = self.realsense2world(self.point2)
        self.point3 = self.realsense2world(self.point3)
        self.point4 = self.realsense2world(self.point4)
        self.point5 = self.realsense2world(self.point5)
        self.point6 = self.realsense2world(self.point6)
        return self.point1, self.point2, self.point3, self.point4, self.point5, self.point6
    
    def projection(self):
        self.point1_mat = np.array([[self.point1[0]*self.point1[2]],[self.point1[1]*self.point1[2]],[self.point1[2]]])
        self.point2_mat = np.array([[self.point2[0]*self.point2[2]],[self.point2[1]*self.point2[2]],[self.point2[2]]])
        self.point3_mat = np.array([[self.point3[0]*self.point3[2]],[self.point3[1]*self.point3[2]],[self.point3[2]]])
        self.point4_mat = np.array([[self.point4[0]*self.point4[2]],[self.point4[1]*self.point4[2]],[self.point4[2]]])
        self.point5_mat = np.array([[self.point5[0]*self.point5[2]],[self.point5[1]*self.point5[2]],[self.point5[2]]])
        self.point6_mat = np.array([[self.point6[0]*self.point6[2]],[self.point6[1]*self.point6[2]],[self.point6[2]]])
        p_mat = np.array([[0,0,1],[-0.00160981,0,0.5177980],[0,-0.00160981,0.390396]])
        mat1 = np.dot(p_mat,self.point1_mat)
        mat2 = np.dot(p_mat,self.point2_mat)
        mat3 = np.dot(p_mat,self.point3_mat)
        mat4 = np.dot(p_mat,self.point4_mat)
        mat5 = np.dot(p_mat,self.point5_mat)
        mat6 = np.dot(p_mat,self.point6_mat)
        self.point1[0], self.point1[1], self.point1[2] = mat1[0][0]*0.001, mat1[1][0]*0.001, mat1[2][0]*0.001
        self.point2[0], self.point2[1], self.point2[2] = mat2[0][0]*0.001, mat2[1][0]*0.001, mat2[2][0]*0.001
        self.point3[0], self.point3[1], self.point3[2] = mat3[0][0]*0.001, mat3[1][0]*0.001, mat3[2][0]*0.001
        self.point4[0], self.point4[1], self.point4[2] = mat4[0][0]*0.001, mat4[1][0]*0.001, mat4[2][0]*0.001
        self.point5[0], self.point5[1], self.point5[2] = mat5[0][0]*0.001, mat5[1][0]*0.001, mat5[2][0]*0.001
        self.point6[0], self.point6[1], self.point6[2] = mat6[0][0]*0.001, mat6[1][0]*0.001, mat6[2][0]*0.001

    def realsense2world(self, point): #좌표계의 변환
        return np.matmul(self.matrix_realsense2world, point) + self.realsense_in_world
    

if __name__ == "__main__":
    point_estimation = HeadposeEstimationModel()

    path = "custum/img_data/check/facial_landmark"
    file_list = os.listdir(path)
    file_list_jpg = [file for file in file_list if file.endswith(".txt")]
    
    title_data = {'a':['index'],'b':['point1'],'c':['point2'],'d':['point3'],'e':['point4'],'f':['point5'],'g':['point6']}
    df1 = pd.DataFrame(title_data)
    df1.to_csv('custum/img_data/check/six_point.csv', mode='a', index=False, header=False)

    for file_name in file_list_jpg:
        facial_landmark = open("custum/img_data/check/facial_landmark/{}.txt".format(file_name[:-4]), 'r')
        facial_landmark_data = facial_landmark.read()
        facial_landmark_data = facial_landmark_data.split('\n')[:-1]
        bin_data = np.load("custum/img_data/check/bin/depth_{}.npy".format(int(file_name[:-4])))
        point1, point2, point3, point4, point5, point6 = point_estimation.forward(depth_img= bin_data ,facial_landmark= facial_landmark_data)
        with open('custum/img_data/check/six_point.csv', 'a') as f:
            f.write(file_name[:-4]+','+str(point1)+','+str(point2)+','+str(point3)+','+str(point4)+','+str(point5)+','+str(point6)+'\n')