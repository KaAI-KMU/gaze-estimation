import csv
import os
import shutil

case_folder = 'case17'
# CSV 파일 경로
csv_file = 'dataset/{}/csv/driver.csv'.format(case_folder)

# JPG 파일이 저장된 디렉토리 경로
jpg_directory = 'dataset/{}/after_gan'.format(case_folder)

remove_folder = 'dataset/remove/{}'.format(case_folder)

os.makedirs(remove_folder, exist_ok=True)

count = 1
# CSV 파일 열기
with open(csv_file, 'r') as file:
    # CSV 데이터 읽기
    reader = csv.reader(file)
    next(reader)  # 헤더 행 건너뛰기

    # 각 행에 대해 작업 수행
    for row in reader:
        confidence = float(row[1])  # confidence 열 값 가져오기
        filename = '{0:06d}.jpg'.format(count)  # JPG 파일 이름 생성

        # confidence 값이 0.8 이하인 경우 JPG 파일 삭제
        if confidence <= 0.8:
            file_path = os.path.join(jpg_directory, filename)
            if os.path.exists(file_path):
                remove_path = os.path.join(remove_folder, filename)
                shutil.move(file_path, remove_path)
                # os.remove(file_path)
                print(f"Deleted file: {filename}")
                
        count+=1
print("Deletion complete.")
