import dlib
import cv2
import os

folder_name = 'check'
path = "custum/img_data/{}".format(folder_name)
img_path = "custum/img_data/{}/rgb_img".format(folder_name)

file_list = os.listdir(img_path)
file_list_jpg = [file for file in file_list if file.endswith(".jpg")]#.png
print(file_list_jpg)
count = 1

# Load the facial landmark predictor from dlib
predictor = dlib.shape_predictor('custum/shape_predictor_68_face_landmarks.dat')

# Load the face detector from dlib
detector = dlib.get_frontal_face_detector()

# Create separate folders to save the face and eyes
face_folder = '{}/faces/'.format(path)
leye_folder = '{}/left_eyes/'.format(path)
reye_folder = '{}/right_eyes/'.format(path)
facial_landmark_folder = '{}/facial_landmark/'.format(path)
headpose_save = '{}/headpose_save/'.format(path)

if not os.path.exists(face_folder):
    os.makedirs(face_folder)
if not os.path.exists(leye_folder):
    os.makedirs(leye_folder)
if not os.path.exists(reye_folder):
    os.makedirs(reye_folder)
if not os.path.exists(facial_landmark_folder):
    os.makedirs(facial_landmark_folder)
if not os.path.exists(headpose_save):
    os.makedirs(headpose_save)

# Loop through each image in the directory
for file_name in file_list_jpg:
    # Load the image
    #img = cv2.imread('custum/img_data/{}'.format(file_name))#rt-gene_test/
    img = cv2.imread('{}/{}'.format(img_path,file_name))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    if len(faces) == 0:
        # Save the file directory to txt file if no face is detected
        with open('{}/facial_landmark/no_face_detected.txt'.format(path), 'a') as f:
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
            face_depth_img = img[y:y+h, x:x+w]

            # Save the face and eyes separately with unique names to avoid overwriting
            face_file = os.path.join(face_folder, os.path.splitext(file_name)[0] + '_face.jpg')#png
            cv2.imwrite(face_file, face_img)

            leye_file = os.path.join(leye_folder, os.path.splitext(file_name)[0] + '_leye.jpg')
            cv2.imwrite(leye_file, img[left_eye[1]:left_eye[3], left_eye[0]:left_eye[2]])

            reye_file = os.path.join(reye_folder, os.path.splitext(file_name)[0] + '_reye.jpg')
            cv2.imwrite(reye_file, img[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2]])

            with open('{}/facial_landmark/{}.txt'.format(path,file_name[:-4]), 'w') as f:
                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    f.write('{},{}\n'.format(x, y))
            
            # 눈, 코, 입, 턱 위치 계산
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
            mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
            mouth_right = (landmarks.part(54).x, landmarks.part(54).y)
            chin = (landmarks.part(8).x, landmarks.part(8).y)

            # 눈 중심 계산
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

            # 머리 방향 계산
            dx = nose_tip[0] - eye_center[0]
            dy = nose_tip[1] - eye_center[1]
            angle = -cv2.fastAtan2(dx, dy)

            # 머리 방향에 화살표 그리기
            cv2.arrowedLine(img, (int(chin[0]), int(chin[1])), (int(chin[0] - 50 * dx / 100), int(chin[1] - 50 * dy / 100)),
                    (0, 0, 255), 2)
        # 이미지 파일에 검출 결과 저장
        cv2.imwrite("{}/%06d.jpg".format(headpose_save) % count, img)
    
        print(count)
    count += 1
