import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
from sklearn.metrics import f1_score

print("Тестирование модели машинного обучения...")
with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv("./test/test_data_trans.csv")
y_test = np.array(pd.read_csv("./test/test_target_trans.csv")).reshape(X_test.shape[0])

y_pred = model.predict(X_test)
print(f"Метрика F1 на тестовой выборке {f1_score(y_pred, y_test)}")

