import torch
import cv2
import numpy as np

# # TCDCN 모델 불러오기
# model_path = "/home/kaai/Tensorflow-TCDCN/train(40)/model-29999.meta"
# model = torch.load(model_path)

# # 이미지 불러오기
# image_path = "custum/img_data/check/rgb_img/000001.jpg"
# image = cv2.imread(image_path)

# # 이미지 전처리
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.resize(image, (128, 128))
# image = np.expand_dims(image, axis=0)
# image = np.expand_dims(image, axis=0)

# # 추론하기
# with torch.no_grad():
#     inputs = torch.from_numpy(image).float()
#     outputs = model(inputs)
#     landmarks = outputs[0].numpy().reshape(-1, 2)

# # 결과 시각화
# for landmark in landmarks:
#     x, y = landmark
#     cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), -1)

# cv2.imshow("Landmarks", image)
# cv2.waitKey(0)
