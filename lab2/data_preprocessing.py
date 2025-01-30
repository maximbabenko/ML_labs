import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

X_train = pd.read_csv("./train/train_data.csv")
X_test = pd.read_csv("./test/test_data.csv")

y_train = pd.read_csv("./train/train_target.csv")
y_test = pd.read_csv("./test/test_target.csv")

print("Преобразование данных...", end="")
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

encoder = OrdinalEncoder()
y_train_transformed = encoder.fit_transform(y_train)
y_test_transformed = encoder.transform(y_test)

pd.DataFrame(X_train_transformed).to_csv("./train/train_data_trans.csv", index=False)
pd.DataFrame(X_test_transformed).to_csv("./test/test_data_trans.csv", index=False)
pd.DataFrame(y_train_transformed).to_csv("./train/train_target_trans.csv", index=False)
pd.DataFrame(y_test_transformed).to_csv("./test/test_target_trans.csv", index=False)

print("Done")