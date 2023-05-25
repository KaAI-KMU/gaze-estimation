import torch
import os
from argparse import ArgumentParser
from train_model import TrainEstimationModel
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import math

root_dir = os.path.dirname(os.path.realpath(__file__))#h5파일이 저장된 디렉토리로 설정

_root_parser = ArgumentParser(add_help=False)
_root_parser.add_argument('--accelerator', choices=['cpu','gpu'], default='gpu',
                        help='gpu to use, can be repeated for mutiple gpus i.e. --gpu 1 --gpu 2')
_root_parser.add_argument('--hdf5_file', type=str,
                        default=os.path.abspath(os.path.join(root_dir, "../KaAI_dataset_3case.hdf5")))#데이터셋이 담긴 h5 파일 지명
_root_parser.add_argument('--dataset', type=str, choices=["KaAI", "other"], default="KaAI")
_root_parser.add_argument('--save_dir', type=str, default='Gaze Estimate Model/custum/checkpoints')#체크포인트 저장할 디렉토리 설정
_root_parser.add_argument('--benchmark', action='store_true', dest="benchmark")
_root_parser.add_argument('--no_benchmark', action='store_false', dest="benchmark")
_root_parser.add_argument('--num_io_workers', default=0, type=int)
_root_parser.add_argument('--k_fold_validation', default=False, type=bool)
_root_parser.add_argument('--accumulate_grad_batches', default=1, type=int)
_root_parser.add_argument('--seed', type=int, default=0)
_root_parser.add_argument('--min_epochs', type=int, default=5, help="Number of Epochs to perform at a minimum")
_root_parser.add_argument('--max_epochs', type=int, default=20,
                        help="Maximum number of epochs to perform; the trainer will Exit after.")
_root_parser.add_argument('--checkpoint',type=list, default=[])
_root_parser.set_defaults(benchmark=False)
_root_parser.set_defaults(augment=True)

_model_parser = TrainEstimationModel.add_model_specific_args(_root_parser)
_hyperparams = _model_parser.parse_args()
_hyperparams = vars(_hyperparams)

_train_subjects = []
_valid_subjects = []
_test_subjects = []

if _hyperparams['dataset'] == "KaAI":
    if _hyperparams['k_fold_validation']:
        _train_subjects.append([1, 2, 8, 10, 3, 4, 7, 9])
        _train_subjects.append([1, 2, 8, 10, 5, 6, 11, 12, 13])
        _train_subjects.append([3, 4, 7, 9, 5, 6, 11, 12, 13])
        # validation set is always subjects 14, 15 and 16
        _valid_subjects.append([0, 14, 15, 16])
        _valid_subjects.append([0, 14, 15, 16])
        _valid_subjects.append([0, 14, 15, 16])
        # test subjects
        _test_subjects.append([5, 6, 11, 12, 13])
        _test_subjects.append([3, 4, 7, 9])
        _test_subjects.append([1, 2, 8, 10])
    else:
        _train_subjects.append([0,1,2])#, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        _train_subjects.append([0,1])
        _valid_subjects.append([1])  # Note that this is a hack and should not be used to get results for papers
        _test_subjects.append([0])


gaze_estimation_model = TrainEstimationModel(hparams=_hyperparams,
                            train_subjects=_train_subjects,
                            validate_subjects=_valid_subjects,
                            test_subjects=_test_subjects)
check_point = torch.load('Gaze Estimate Model/custum/checkpoints/vgg16/epoch=12-val_loss=0.029.ckpt')
gaze_estimation_model.load_state_dict(check_point['state_dict'])
gaze_estimation_model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

face_image = cv2.imread('Gaze Estimate Model/custum/img_data/check/000043_face.jpg')
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
face_image_pil = Image.fromarray(face_image)
face_image_tensor = transform(face_image_pil)
left_eye_image = cv2.imread('Gaze Estimate Model/custum/img_data/check/000043_leye.jpg')
left_eye_image = cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2RGB)
left_eye_image_pil = Image.fromarray(left_eye_image)
left_eye_image_tensor = transform(left_eye_image_pil)
right_eye_image = cv2.imread('Gaze Estimate Model/custum/img_data/check/000043_reye.jpg')
right_eye_image = cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2RGB)
right_eye_image_pil = Image.fromarray(right_eye_image)
right_eye_image_tensor = transform(right_eye_image_pil)

csv_file = open('Gaze Estimate Model/custum/img_data/check/driver.csv')
csv_data = csv_file.read().split('\n')[:-1]
csv_slice_data = [[index.split(',')]for index in csv_data]
head_pose = np.array([float(csv_slice_data[43][:][0][2]),float(csv_slice_data[43][:][0][3]),float(csv_slice_data[43][:][0][4])])

headpose_angle = np.array([math.atan2(math.sqrt(head_pose[0]*head_pose[0]+head_pose[1]*head_pose[1]),head_pose[2]),math.atan2(head_pose[1],head_pose[0])])

eye_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

with torch.no_grad():
    output = gaze_estimation_model(left_eye_image_tensor.unsqueeze(0).float(), right_eye_image_tensor.unsqueeze(0).float(), torch.from_numpy(headpose_angle).unsqueeze(0).float())

theta = output[:, 0]
phi = output[:, 1]
sin_theta = torch.sin(theta)
cos_theta = torch.cos(theta)
sin_phi = torch.sin(phi)
cos_phi = torch.cos(phi)

x = sin_theta * cos_phi
y = sin_theta * sin_phi
z = cos_theta

gaze_vector = torch.stack((x, y, z), dim=1)
print('Prediction:', gaze_vector)