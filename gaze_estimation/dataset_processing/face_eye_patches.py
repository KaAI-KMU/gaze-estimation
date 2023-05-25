import dlib
import cv2
import os
import math
from ..model.estimation_headpose import HeadposeEstimationModel

path = "custum/img_data/case2/color_img"
depth_path = "custum/img_data/case2/depth_img"
bin_path = "custum/img_data/case2/bin"
save_path = "custum/img_data/case2"

file_list = os.listdir(path)
file_list_jpg = [file for file in file_list if file.endswith(".jpg")]#.png
count = 1

# Load the facial landmark predictor from dlib
predictor = dlib.shape_predictor('custum/shape_predictor_68_face_landmarks.dat')

# Load the face detector from dlib
detector = dlib.get_frontal_face_detector()

# Create separate folders to save the face and eyes
face_folder = '{}/faces/'.format(save_path)
face_depth_folder = '{}/faces_depth/'.format(save_path)
leye_folder = '{}/left_eyes/'.format(save_path)
reye_folder = '{}/right_eyes/'.format(save_path)
facial_landmark_folder = '{}/facial_landmark/'.format(save_path)

if not os.path.exists(face_folder):
    os.makedirs(face_folder)
if not os.path.exists(face_depth_folder):
    os.makedirs(face_depth_folder)
if not os.path.exists(leye_folder):
    os.makedirs(leye_folder)
if not os.path.exists(reye_folder):
    os.makedirs(reye_folder)
if not os.path.exists(facial_landmark_folder):
    os.makedirs(facial_landmark_folder)

# Loop through each image in the directory
for file_name in file_list_jpg:
    # Load the image
    #img = cv2.imread('custum/img_data/{}'.format(file_name))#rt-gene_test/
    img = cv2.imread('{}/{}'.format(path,file_name))
    img_depth = cv2.imread('{}/{}'.format(depth_path,file_name), cv2.CV_16UC1 )#cv2.IMREAD_UNCHANGED

    estimation_headpose = HeadposeEstimationModel()

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    if len(faces) == 0:
        # Save the file directory to txt file if no face is detected
        with open('{}/facial_landmark/no_face_detected.txt'.format(save_path), 'a') as f:
            f.write(file_name+'\n')
        # os.remove('{}/{}'.format(path,file_name))
        # os.remove('{}/{}'.format(depth_path,file_name))
    else:
        # Loop through each face and extract the facial landmarks and save the face and eyes separately
        for face in faces:
            # Get the facial landmarks
            landmarks = predictor(gray, face)

            left_eye_x = (landmarks.part(39).x-landmarks.part(36).x)//2
            left_eye_y = (landmarks.part(41).y-landmarks.part(37).y)//2

            right_eye_x = (landmarks.part(45).x-landmarks.part(42).x)//2
            right_eye_y = (landmarks.part(47).y-landmarks.part(43).y)//2

            # Extract the coordinates of the left and right eyes
            left_eye = (landmarks.part(36).x - left_eye_x, landmarks.part(37).y - left_eye_y, landmarks.part(39).x + left_eye_x, landmarks.part(41).y + left_eye_y)
            right_eye = (landmarks.part(42).x - right_eye_x, landmarks.part(43).y - right_eye_y, landmarks.part(45).x + right_eye_x, landmarks.part(47).y + right_eye_y)

            # Extract the face from the image
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_img = img[y:y+h, x:x+w]
            face_depth_img = img_depth[y:y+h, x:x+w]

            # Save the face and eyes separately with unique names to avoid overwriting
            face_file = os.path.join(face_folder, os.path.splitext(file_name)[0] + '_face.jpg')#png
            cv2.imwrite(face_file, face_img)

            face_depth_file = os.path.join(face_depth_folder, os.path.splitext(file_name)[0] + '_face_depth.jpg')#png
            cv2.imwrite(face_depth_file, face_depth_img)

            leye_file = os.path.join(leye_folder, os.path.splitext(file_name)[0] + '_leye.jpg')
            cv2.imwrite(leye_file, img[left_eye[1]:left_eye[3], left_eye[0]:left_eye[2]])

            reye_file = os.path.join(reye_folder, os.path.splitext(file_name)[0] + '_reye.jpg')
            cv2.imwrite(reye_file, img[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2]])

            with open('{}/facial_landmark/{}.txt'.format(save_path,file_name[:-4]), 'w') as f:
                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    f.write('{},{}\n'.format(x, y))
    
        print(count)
        # facial_landmark = open("custum/img_data/rgb_img/facial_landmark/{}.txt".format(file_name[:-4]), 'r')
        # facial_landmark_data = facial_landmark.read()
        # facial_landmark_data = facial_landmark_data.split('\n')[:-1]
        # f = open("custum/img_data/rgb_img/bin/depth_{}.bin".format(count),'r')
        # bin_data = f.readlines()
        # for index in range(len(bin_data)):
        #     bin_data[index] = bin_data[index].split(' ')[:-1]
        # estimation_headpose_value = estimation_headpose.forward('{}/{}'.format(path,file_name), bin_data ,facial_landmark_data)
        # f.close
        # csv_max = os.listdir('custum/img_data/rgb_img/color_img')
        # label = open("custum/img_data/rgb_img/csv/driver.csv", 'r')
        # label_data = label.read()
        # label_data = label_data.split('\n')
        # label_data = label_data[:len(csv_max)+1]
        # label_slice_data = [[index.split(',')]for index in label_data]

        # with open("custum/img_data/rgb_img/csv/estimate_headpose.csv",'a') as hf:
        #     hf.write(str(count) + ',' + str(estimation_headpose_value[0]) + ',' + str(estimation_headpose_value[1]) + ',' + str(estimation_headpose_value[2]) + ',' + 
        #               str(label_slice_data[count][0][2]) + ',' + str(label_slice_data[count][0][3]) + ',' + str(label_slice_data[count][0][4]) + '\n')
        # print(estimation_headpose_value)
    count += 1