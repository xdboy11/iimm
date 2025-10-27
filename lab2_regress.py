import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Обучающая выборка:
#  MSE: 0.0299
#  MAE: 0.1350
#Тестовая выборка:
#  MSE: 0.0314
#  RMSE: 0.1773
#  MAE: 0.1398

#Обучающая выборка:
#  MSE: 187.9332
#  MAE: 10.6775
#Тестовая выборка:
#  MSE: 196.2622
#  RMSE: 14.0094 - x
#  MAE: 10.9933

#Обучающая выборка:
#  MSE: 186.3516
#  MAE: 10.6657
#Тестовая выборка:
#  MSE: 196.1819
#  RMSE: 14.0065
#  MAE: 11.0474

#0.2%

pd.set_option('display.max_columns', None)
df = pd.read_csv("./data/processed_train.csv")
df2 = pd.read_csv("./data/processed_train2.csv")

df['Age'] = df2['Age']

df = df.drop(['Cabin', 'Name'], axis = 1)

X = df.drop('Age', axis=1)
y = df['Age']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_train = linear_model.predict(X_train)
y_pred_test = linear_model.predict(X_test)

print("-----Оценка-----")

mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

print(f"\nОбучающая выборка:")
print(f"  MSE: {mse_train:.4f}")
print(f"  MAE: {mae_train:.4f}")

print(f"Тестовая выборка:")
print(f"  MSE: {mse_test:.4f}")
print(f"  RMSE: {rmse_test:.4f}")
print(f"  MAE: {mae_test:.4f}")