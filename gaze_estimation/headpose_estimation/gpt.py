import cv2
import dlib
import numpy as np

# 68개의 facial landmark 좌표를 저장한 .dat 파일 경로
predictor_path = "shape_predictor_68_face_landmarks.dat"

# dlib의 얼굴 인식기와 68개의 facial landmark 예측기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 카메라 내부 파라미터
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

# 68개의 facial landmark와 depth 정보를 이용하여 얼굴의 3차원 좌표를 계산하는 함수
def get_3d_points(image_points, depth):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    image_points = np.concatenate([image_points, np.ones((image_points.shape[0], 1))], axis=1)
    X = (image_points[:, 0] - cx) * depth / fx
    Y = (image_points[:, 1] - cy) * depth / fy
    Z = depth

    return np.vstack([X, Y, Z]).transpose()

# PnP 알고리즘을 이용하여 얼굴의 위치와 방향을 추정하는 함수
def estimate_head_pose(image, landmarks):
    image_points = np.array([
        landmarks[30], # nose tip
        landmarks[8],  # chin
        landmarks[36], # left eye corner
        landmarks[45], # right eye corner
        landmarks[48], # left mouth corner
        landmarks[54]  # right mouth corner
    ], dtype="double")

    # depth 정보는 예시로 50으로 설정
    depth = 50

    # 68개의 facial landmark와 depth 정보를 이용하여 얼굴의 3차원 좌표를 계산
    model_points = get_3d_points(image_points, depth)

    # PnP 알고리즘을 이용하여 얼굴의 위치와 방향 추정
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, K, np.zeros((4, 1)))

    # 회전 벡터를 이용하여 회전 각도 추정
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack([rvec_matrix, translation_vector])
    euler_angles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]

    return euler_angles

# 이미지 불러오기
image = cv2.imread("image.jpg")

# 얼굴 인식
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)

# 얼굴이 인식되었을 때
if len(rects) > 0:
    # 인식된 얼굴마다 처리
    for rect in rects:
        # 얼굴에서 68개의 facial landmark 추출
        landmarks = np.array([[p.x, p.y] for p in predictor(gray, rect).parts()])

        # 추출된 랜드마크를 이용하여 head pose 추정
        euler_angles = estimate_head_pose(image, landmarks)

        # 결과 출력
        print("Yaw: {:.2f}, Pitch: {:.2f}, Roll: {:.2f}".format(euler_angles[1], euler_angles[0], euler_angles[2]))

        # 랜드마크와 head pose 정보를 이미지에 그리기
        for point in landmarks:
            cv2.circle(image, tuple(point), 1, (0, 255, 0), -1)
        draw_axes(image, euler_angles)

    # 결과 이미지 출력
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()