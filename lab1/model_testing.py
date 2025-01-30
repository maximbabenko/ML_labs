import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import pickle
from sklearn.metrics import mean_squared_error as mse

print("Тестирование модели машинного обучения...")
with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv("./test/test_data_trans.csv")
y_test = np.array(pd.read_csv("./test/test_target.csv")).reshape(X_test.shape[0])

y_pred = model.predict(X_test)
print(f"Метрика MSE на тестовой выборке {mse(y_pred, y_test)}")

