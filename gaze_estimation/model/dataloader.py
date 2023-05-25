import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import math
import re
import torch

class EstimationHdf5DatasetMyDataset(Dataset):
    def __init__(self, h5_file, subject_list=None, transform=None):
        self._h5_file = h5_file
        self._transform = transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((256,256), Image.BICUBIC),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        _wanted_subjects = ["case{}".format(_i+1) for _i in subject_list]

        for grp_s_n in tqdm(_wanted_subjects, desc="Loading subject metadata..."):  # subjects
            for grp_i_n, grp_i in h5_file[grp_s_n].items():  # images
                if "left_eye" in grp_i.keys() and "right_eye" in grp_i.keys():
                    left_dataset = grp_i["left_eye"]
                    right_datset = grp_i['right_eye']

                    assert len(left_dataset) == len(right_datset), "Dataset left/right images aren't equal length"
                    for _i in left_dataset.keys():
                        self._subject_labels.append(["/" + grp_s_n + "/" + grp_i_n, _i])

    def __getitem__(self, index):
        _sample = self._subject_labels[index]
        assert type(_sample[0]) == str, "Sample not found at index {}".format(index)
        _left_img = self._h5_file[_sample[0] + "/left_eye"][_sample[1]][()]
        _right_img = self._h5_file[_sample[0] + "/right_eye"][_sample[1]][()]
        label_data = self._h5_file["{}/text/label/{}_label".format(_sample[0][:6],_sample[0][1:6])][()]#int(_sample[1][:-4])
        label_string = str(label_data[int(_sample[1][:-4])][0][2:5])[3:-2]#x,y,z 따로면 수정해야함.
        pattern = r'[-+]?\d*\.\d+|\d+'
        label_list = [float(n) for n in re.findall(pattern, label_string)]
        _groud_truth_headpose_vector = np.array([float(label_list[0]),float(label_list[1]),float(label_list[2])])#label_data[int(_sample[1][:-4])][0][2:4].astype(np.float32)
        _ground_truth_headpose = np.array([math.atan2(math.sqrt(_groud_truth_headpose_vector[0]*_groud_truth_headpose_vector[0]+_groud_truth_headpose_vector[1]*_groud_truth_headpose_vector[1]),_groud_truth_headpose_vector[2]),math.atan2(_groud_truth_headpose_vector[1],_groud_truth_headpose_vector[0])])
        gaze_string = str(label_data[int(_sample[1][:-4])][0][14:17])[3:-2]
        gaze_list = [float(n) for n in re.findall(pattern, gaze_string)]
        _ground_truth_gaze_vector = np.array([float(gaze_list[0]),float(gaze_list[1]),float(gaze_list[2])])
        _ground_truth_gaze = np.array([math.atan2(math.sqrt(_ground_truth_gaze_vector[0]*_ground_truth_gaze_vector[0]+_ground_truth_gaze_vector[1]*_ground_truth_gaze_vector[1]),_ground_truth_gaze_vector[2]),math.atan2(_ground_truth_gaze_vector[1],_ground_truth_gaze_vector[0])])
        landmark_data = self._h5_file["{}/text/facial_landmark/{}_facial_landmark".format(_sample[0][:6],_sample[1][:-4])][()]
        _landmark_data = np.array([np.array(item.decode("utf-8").split(","), dtype=np.int32) for item in landmark_data])

        # Load data and get label
        _transformed_left = self._transform(Image.fromarray(_left_img, 'RGB'))
        _transformed_right = self._transform(Image.fromarray(_right_img, 'RGB'))

        _ground_truth_headpose = torch.from_numpy(_ground_truth_headpose).float()
        _ground_truth_gaze = torch.from_numpy(_ground_truth_gaze).float()
        _landmark_data = torch.from_numpy(_landmark_data).float()

        return _transformed_left, _transformed_right, _ground_truth_headpose, _ground_truth_gaze, _landmark_data

    def __len__(self):
        return len(self._subject_labels)

class EstimationFileDataset(Dataset):
    def __init__(self, img_dir, depth_dir, landmark_dir, left_eye_dir, right_eye_dir, label_dir, face_transform=None, eye_transform=None):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.landmark_dir = landmark_dir
        self.left_eye_dir = left_eye_dir
        self.right_eye_dir = right_eye_dir
        self.label_dir = label_dir

        self._face_transform = face_transform
        self._eye_transform = eye_transform
        if self._face_transform is None:
            self._face_transform = transforms.Compose([transforms.Resize((256,256), Image.BICUBIC),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if self._eye_transform is None:
            self._eye_transform = transforms.Compose([transforms.Resize((64,64), Image.BICUBIC),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        # load image
        img_path = os.path.join(self.img_dir, f"00000{idx:06d}_face.jpg")
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        # load depth image
        depth_path = os.path.join(self.depth_dir, f"00000{idx:06d}_depth.jpg")
        depth = Image.open(depth_path).convert('L')
        depth = np.array(depth)

        # load landmark data
        landmark_path = os.path.join(self.landmark_dir, f"00000{idx:06d}.txt")
        with open(landmark_path, "r") as f:
            landmark = f.read().split(',')
        landmark = landmark[:-1]

        # load left eye image
        left_eye_path = os.path.join(self.left_eye_dir, f"00000{idx:06d}_leye.jpg")
        left_eye = Image.open(left_eye_path).convert('RGB')
        left_eye = np.array(left_eye)

        # load right eye image
        right_eye_path = os.path.join(self.right_eye_dir, f"00000{idx:06d}_reye.jpg")
        right_eye = Image.open(right_eye_path).convert('RGB')
        right_eye = np.array(right_eye)

        # load label, no fix
        label_path = os.path.join(self.label_dir, f"00000{idx:06d}_label.txt")
        with open(label_path, "r") as f:
            label = f.read().strip()

        return {
            'image': img,
            'depth': depth,
            'landmark': landmark,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'label': label
        }