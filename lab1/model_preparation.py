import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
import os
import pickle

X_train = pd.read_csv("./train/train_data_trans.csv")
y_train = np.array(pd.read_csv("./train/train_target.csv")).reshape(X_train.shape[0])

print("Обучение модели машинного обучения...")
regressor = RandomForestRegressor(n_estimators=250, max_depth=25, n_jobs=-1, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_train)
print(f"Метрика MSE на тренировочной выборке {mse(y_pred, y_train)}")

if not os.path.exists('model'):
    os.makedirs('model')

with open('./model/model.pkl', 'wb') as f:
    pickle.dump(regressor, f)

