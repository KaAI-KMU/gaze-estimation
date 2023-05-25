from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

def extract_facial_landmark(image, detector, predictor):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 1)
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)
	cv2.waitKey(0)

# def crop_face(image):
# 	blob = cv2.dnn.blobFromImage(
#         frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0),
#     )
# 	detector.setInput(blob)
# 	face_detections = detector.forward()

# 	for i in range(0,face_detections.shape[2]):
# 		confidence = face_detections[0, 0, i, 2]
# 		if confidence > 0.5:
# 			box = face_detections[0, 0, i, 3:7] * np.array([width, height, width, height])
#         	(x1, y1, x2, y2) = box.astype("int")

# 			cv2.imshow("original image", frame)
# 			resized = crop(
# 				frame,
# 				torch.Tensor([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]),
# 				1.5,
# 				tuple(input_size),
#         	)
# 			resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         	img = resized.astype(np.float32) / 255.0

# 			normalized_img = (img - mean) / std



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

index = ALL

image = cv2.imread('./img_data/000004.jpg')
image = imutils.resize(image, width=500)
extract_facial_landmark(image, detector, predictor)
