import h5py
import numpy as np
from tqdm import tqdm

# path = 'custum/img_data/rgb_img/depth_img/000001.jpg'
path = 'Gaze Estimate Model/custum/KaAI_dataset_test.hdf5'

h5_file = h5py.File(path, 'r')

_wanted_subjects = ['case3']
_subject_labels = []

for grp_s_n in tqdm(_wanted_subjects, desc="Loading subject metadata..."):  # subjects
    for grp_i_n, grp_i in h5_file[grp_s_n].items():  # images
        if "left_eye" in grp_i.keys() and "right_eye" in grp_i.keys():
            left_dataset = grp_i["left_eye"]
            right_datset = grp_i['right_eye']

            assert len(left_dataset) == len(right_datset), "Dataset left/right images aren't equal length"
            for _i in left_dataset.keys():
                _subject_labels.append(["/" + grp_s_n + "/" + grp_i_n, _i])

_sample = _subject_labels[0]
assert type(_sample[0]) == str, "Sample not found at index {}".format(0)
_left_img = h5_file[_sample[0] + "/left_eye"][_sample[1]][()]
_right_img = h5_file[_sample[0] + "/right_eye"][_sample[1]][()]

# label h5 파일에서 불러오는 거 부터 합시다!

label_data = h5_file["{}/text/label/{}_label".format(_sample[0][:6],_sample[0][1:6])][()]#int(_sample[1][:-4])
_groud_truth_headpose = str(label_data[int(_sample[1][:-4])][0][2])[3:-2].split('  ')
_ground_truth_gaze = label_data[int(_sample[1][:-4])][0][14:16].astype(np.float32)

landmark_data = h5_file["{}/text/facial_landmark/{}_facial_landmark".format(_sample[0][:6],_sample[1][:-4])][()]


# for grp_s_n in tqdm(_wanted_subjects, desc="Loading subject metadata..."):  # subjects
#     for number in h5_file[grp_s_n].keys(): # folder number in case
#         if "left_eye" in number.keys() and "right_eye" in number.keys() and "label" in number.keys() and "face_color" in number.keys() and "face_depth" in number.keys():
#             left_dataset = number['left_eye']
#             right_dataset = number['right_eye']

#             assert len(left_dataset) == len(right_dataset), "Dataset left/right images aren't equal length"
#             for _i in range(len(left_dataset)):
#                 _subject_labels.append(["/" + grp_s_n + "/" + number, _i])