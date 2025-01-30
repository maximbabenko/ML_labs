from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import os

print("Создание набора данных...", end="")

X, y = make_regression(n_samples=10000, 
                       n_features=8, 
                       n_informative=5, 
                       noise=15, 
                       random_state=42, 
                       bias=10)
X = pd.DataFrame(X)
y = pd.Series(y)
print("Done")

if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

print("Сохранение набора данных...", end="")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pd.DataFrame(X_train).to_csv("./train/train_data.csv", index=False)
pd.DataFrame(y_train).to_csv("./train/train_target.csv", index=False)
pd.DataFrame(X_test).to_csv("./test/test_data.csv", index=False)
pd.DataFrame(y_test).to_csv("./test/test_target.csv", index=False)
print("Done")

