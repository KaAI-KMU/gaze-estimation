from sklearn.svm import SVR
import numpy as np

X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 입력 데이터 (세 점의 3차원 위치)
y_train = np.array([10, 20, 30])  # 출력 데이터 (세 점의 위치)

# SVM 회귀 모델 초기화
svm_regressor = SVR(kernel='linear', C=1.0)  # 선형 커널을 사용하는 SVM 회귀 모델

# 모델 학습
svm_regressor.fit(X_train, y_train)

# 예측
X_test = np.array([[2, 3, 4]])  # 예측할 입력 데이터 (새로운 3차원 위치)
y_pred = svm_regressor.predict(X_test)  # 입력 데이터에 대한 출력 값 예측

print("예측된 출력 값:", y_pred)