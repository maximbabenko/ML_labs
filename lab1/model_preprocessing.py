import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("./train/train_data.csv")
X_test = pd.read_csv("./test/test_data.csv")

print("Преобразование данных...", end="")
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

pd.DataFrame(X_train_transformed).to_csv("./train/train_data_trans.csv", index=False)
pd.DataFrame(X_test_transformed).to_csv("./test/test_data_trans.csv", index=False)
print("Done")
