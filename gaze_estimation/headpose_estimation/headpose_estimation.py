import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import logging

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('custum/img_data/check/head_pose_prediction.log')
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

df = pd.read_csv('/home/kaai/Gaze Estimate Model/custum/img_data/check/six_point.csv')
df2 = pd.read_csv('/home/kaai/Gaze Estimate Model/custum/img_data/check/csv/driver.csv')

file_name_list = df['index'].tolist()

point1_list = df['point1'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float)).tolist()
point2_list = df['point2'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float)).tolist()
point3_list = df['point3'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float)).tolist()
point4_list = df['point4'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float)).tolist()
point5_list = df['point5'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float)).tolist()
point6_list = df['point6'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float)).tolist()

X_train = np.array([np.concatenate((point1_list[i], point2_list[i], point3_list[i], point4_list[i], point5_list[i], point6_list[i])) for i in range(len(df))])

head_vector = df2.loc[file_name_list, ['head_vec_x', 'head_vec_y', 'head_vec_z']].values

X_train, X_test, y_train, y_test = train_test_split(X_train, head_vector, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

y_train_norm = scaler.fit_transform(y_train)
y_test_norm = scaler.transform(y_test)

######################## MLP #############################################################
input_dim = X_train_norm.shape[1]  # 세 점의 3차원 값 (3 * 3)
output_dim = y_train_norm.shape[1]  # 3차원 벡터의 차원 수
mlp_model = MLP(input_dim, output_dim)

optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)  # Adam optimizer 사용
criterion = nn.MSELoss()

model_name = 'Head_pose_prediction_model'
epochs = 1000
for epoch in range(epochs):
    # Forward 연산
    outputs = mlp_model(torch.Tensor(X_train_norm))
    
    # 손실 계산
    loss = criterion(outputs, torch.Tensor(y_train_norm))
    
    # 역전파 및 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 학습 중간에 손실 출력 등의 필요한 로그 기록
    logger.info('Model "{}" - Epoch: {}'.format(model_name, epochs))
    logger.info(f'Loss value :: {loss.item()}')
    logger.debug('Model "{}" Debug Message'.format(model_name))
    logger.warning('Model "{}" Emergency Message'.format(model_name))
    logger.error('Model "{}" Error Message'.format(model_name))

with torch.no_grad():
    test_outputs = mlp_model(torch.Tensor(X_test_norm))

y_pred_norm = test_outputs.detach().numpy()
y_pred = scaler.inverse_transform(y_pred_norm)
print('Prediction:',y_pred)

save_title_data = {'a':['original_x'],'b':['original_y'],'c':['original_z'],'d':['prediction_x'],'e':['prediction_y'],'f':['prediction_z']}
save_title_data_frame = pd.DataFrame(save_title_data)
save_title_data_frame.to_csv('custum/img_data/check/prediction.csv', mode='a', index=False, header=False)

for test, pred in zip(y_test,y_pred):
    save_data = {'a':[test[0]],'b':[test[1]],'c':[test[2]],'d':[pred[0]],'e':[pred[1]],'f':[pred[2]]}
    save_data_frame = pd.DataFrame(save_data)
    save_data_frame.to_csv('custum/img_data/check/prediction.csv', mode='a', index=False, header=False)
#########################################################################################

####################### SVM #############################################################
# svm_regressor_x = SVR(kernel='poly')
# svm_regressor_y = SVR(kernel='poly')
# svm_regressor_z = SVR(kernel='poly')

# svm_regressor_x.fit(X_train, y_train[:, 0])
# svm_regressor_y.fit(X_train, y_train[:, 1])
# svm_regressor_z.fit(X_train, y_train[:, 2])
#########################################################################################

# y_pred_x = svm_regressor_x.predict(X_test)
# y_pred_y = svm_regressor_y.predict(X_test)
# # y_pred_z = svm_regressor_z.predict(X_test)

# y_pred = np.column_stack((y_pred_x, y_pred_y, y_pred_z))

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print('Prediction:',y_pred)
# print("Mean Squared Error (MSE): ", mse)
# print("R-squared (R2): ", r2)