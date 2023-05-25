############# 종류에 따른 분류###################
import h5py
import numpy as np
import os
from PIL import Image

folder_list=['case1','case2','case3']
h5_file = h5py.File('Gaze Estimate Model/custum/KaAI_dataset_3case.hdf5', 'w') #'a'

for folder in folder_list:
    orginal_dir = "Gaze Estimate Model/custum/img_data/{}/rgb_img".format(folder)
    img_dir="Gaze Estimate Model/custum/img_data/{}/faces".format(folder)
    depth_dir='Gaze Estimate Model/custum/img_data/{}/faces_depth'.format(folder)
    landmark_dir='Gaze Estimate Model/custum/img_data/{}/facial_landmark'.format(folder)
    left_eye_dir='Gaze Estimate Model/custum/img_data/{}/left_eyes'.format(folder)
    right_eye_dir='Gaze Estimate Model/custum/img_data/{}/right_eyes'.format(folder)
    label_dir='Gaze Estimate Model/custum/img_data/{}/csv/'.format(folder)

    h5_file.create_group("{}".format(folder))

    file_list = os.listdir(img_dir)
    csv_max = os.listdir(orginal_dir)

    h5_file.create_group("{}/text/label".format(folder))
    h5_file.create_group("{}/image/left_eye".format(folder))
    h5_file.create_group("{}/image/right_eye".format(folder))
    h5_file.create_group("{}/image/face_color".format(folder))
    h5_file.create_group("{}/image/face_depth".format(folder))
    h5_file.create_group("{}/text/facial_landmark".format(folder))

    label = open("{}/driver.csv".format(label_dir), 'r')
    label_data = label.read()
    label_data = label_data.split('\n')
    label_data = label_data[:len(csv_max)+1]
    label_slice_data = [[index.split(',')]for index in label_data]
    h5_file.create_dataset("{}/text/label/{}_label".format(folder,folder), data=label_slice_data)

    for i in file_list:
        i = i[0:6]

        rgb_img = Image.open('{}/{}_face.jpg'.format(img_dir,i))
        h5_file.create_dataset('{}/image/face_color/{}.jpg'.format(folder,i), data=rgb_img)
        
        depth_img = Image.open('{}/{}_face_depth.jpg'.format(depth_dir,i))
        depth_img = np.asarray(depth_img)
        h5_file.create_dataset('{}/image/face_depth/{}.jpg'.format(folder,i), data=depth_img)

        left_eye_img = Image.open('{}/{}_leye.jpg'.format(left_eye_dir,i))
        h5_file.create_dataset('{}/image/left_eye/{}.jpg'.format(folder,i), data=left_eye_img)

        right_eye_img = Image.open('{}/{}_reye.jpg'.format(right_eye_dir,i))
        h5_file.create_dataset('{}/image/right_eye/{}.jpg'.format(folder,i), data=right_eye_img)

        facial_landmark = open("{}/{}.txt".format(landmark_dir,i), 'r')
        facial_landmark_data = facial_landmark.read()
        facial_landmark_data = facial_landmark_data.split('\n')[:-1]
        h5_file.create_dataset("{}/text/facial_landmark/{}_facial_landmark".format(folder,i), data=facial_landmark_data)

#Case의 경우 사람에 따라 숫자가 달라지고 해당 Case안의 그룹에는 그 순간에 따른 번호가 적혀있다. 또한 해당 번호에는 그 상황에서의 이미지 혹은 label의 정보를 담고 있다.
#현재 코드에는 구현되어 있지 않지만 Case마다 다르게 할 수 있게 .format()을 활용하여 코드를 변형시킬 필요가 있다.

##############순간에 따른 분류#########################
# import h5py
# import numpy as np
# import os
# from PIL import Image

# orginal_dir = "custum/img_data/rgb_img/color_img"
# img_dir="custum/img_data/rgb_img/faces"
# depth_dir='custum/img_data/rgb_img/faces_depth'
# landmark_dir='custum/img_data/rgb_img/facial_landmark'
# left_eye_dir='custum/img_data/rgb_img/left_eyes'
# right_eye_dir='custum/img_data/rgb_img/right_eyes'
# label_dir='custum/img_data/rgb_img/csv/'

# h5_file = h5py.File('custum/KaAI_dataset.hdf5', 'w') #'a'

# h5_file.create_group("Case1")

# file_list = os.listdir(img_dir)
# csv_max = os.listdir(orginal_dir)

# h5_file.create_group("Case1/label")

# label = open("{}/driver.csv".format(label_dir), 'r')
# label_data = label.read()
# label_data = label_data.split('\n')
# label_data = label_data[:len(csv_max)+1]
# label_slice_data = [[index.split(',')]for index in label_data]
# h5_file.create_dataset("Case1/label/Case1_label", data=label_slice_data)

# for i in file_list:
#     i = i[0:6]
#     h5_file.create_group("Case1/{}/left_eye".format(i))
#     h5_file.create_group("Case1/{}/right_eye".format(i))
#     h5_file.create_group("Case1/{}/face_color".format(i))
#     h5_file.create_group("Case1/{}/face_depth".format(i))
#     h5_file.create_group("Case1/{}/facial_landmark".format(i))

#     rgb_img = Image.open('{}/{}_face.jpg'.format(img_dir,i))
#     h5_file.create_dataset('Case1/{}/face_color/{}.jpg'.format(i,i), data=rgb_img)
    
#     depth_img = Image.open('{}/{}_face_depth.jpg'.format(depth_dir,i))
#     depth_img = np.asarray(depth_img)
#     h5_file.create_dataset('Case1/{}/face_depth/{}.jpg'.format(i,i), data=depth_img)

#     left_eye_img = Image.open('{}/{}_leye.jpg'.format(left_eye_dir,i))
#     h5_file.create_dataset('Case1/{}/left_eye/{}.jpg'.format(i,i), data=left_eye_img)

#     right_eye_img = Image.open('{}/{}_reye.jpg'.format(right_eye_dir,i))
#     h5_file.create_dataset('Case1/{}/right_eye/{}.jpg'.format(i,i), data=right_eye_img)

#     facial_landmark = open("{}/{}.txt".format(landmark_dir,i), 'r')
#     facial_landmark_data = facial_landmark.read()
#     facial_landmark_data = facial_landmark_data.split('\n')[:-1]
#     h5_file.create_dataset("Case1/{}/facial_landmark/{}_facial_landmark".format(i,i), data=facial_landmark_data)