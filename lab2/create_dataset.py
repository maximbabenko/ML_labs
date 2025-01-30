import os
import openml
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset
from sklearn.model_selection import train_test_split

print("Скачивание набора данных...", end="")
dataset = get_dataset(37)
print("Done!")

print(f"Датасет называется '{dataset.name}', целевой признак - '{dataset.default_target_attribute}'")
print(f"URL: {dataset.url}")

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

# Создание папок train и test
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
print("Done!")

