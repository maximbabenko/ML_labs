import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import os
import pickle

X_train = pd.read_csv("./train/train_data_trans.csv")
y_train = np.array(pd.read_csv("./train/train_target_trans.csv")).reshape(X_train.shape[0])

print("Обучение модели машинного обучения...")
classifier = RandomForestClassifier(n_estimators=150, max_depth=15, n_jobs=-1, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_train)
print(f"Метрика F1 на тренировочной выборке {f1_score(y_pred, y_train)}")

if not os.path.exists('model'):
    os.makedirs('model')

with open('./model/model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

